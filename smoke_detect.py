#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_detect.py - Deteccion de humo en camaras IP (RTSP / ISAPI).

Metodo: analisis de opacidad por contraste contra zonas de referencia limpias.
No usa modelos entrenados: compara la luminancia y la saturacion de la zona
donde sale la pluma contra zonas testigo de la misma escena que nunca deben
tener humo. Esa diferencia relativa es inmune a cambios de luz, clima y trafico.

Ventaja: funciona sin GPU, sin dataset y sin reentrenar cuando cambia la escena.
Todo lo que hay que hacer es ajustar zonas (ROI) y calibrar dos puntos.

Uso:
    python smoke_detect.py --config config.json --snapshot
    python smoke_detect.py --config config.json --measure
    python smoke_detect.py --config config.json --image foto.jpg
    python smoke_detect.py --calibrate sin_humo.jpg con_humo.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# En Windows la consola suele ser cp1252 y truena al imprimir acentos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --------------------------- Configuracion ---------------------------

DEFAULT_CONFIG = {
    "camera": {
        "rtsp_url": "",
        "isapi_snapshot_url": "",
        "username_env": "CAMERA_USER",
        "password_env": "CAMERA_PASS",
        "timeout": 8,
        "burst_frames": 5,
        "burst_interval": 0.9,
    },
    "zones": {
        "reference_size": [2688, 1520],
        "plume": [1168, 118, 1239, 525],
        "reference_a": [1769, 118, 1874, 525],
        "reference_b": [1008, 42, 1876, 118],
    },
    "calibration": {
        "offset": -15.671429,
        "slope": 1.257143,
        "weights": {"luma_gain": 0.5, "sat_loss_ref_a": 0.4, "sat_loss_ref_b": 0.5},
    },
    "classification": {
        "clear_below": 5.0,
        "gray_from": 40.0,
        "dark_from": 70.0,
    },
    "night": {"enabled": True, "mean_luma_below": 85.0},
    "output": {"dir": "captures", "save_annotated": True},
}


def load_config(path: str | None) -> dict:
    """Carga el JSON y lo fusiona sobre los valores por defecto."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"No existe la configuracion: {p}")
    deep_update(cfg, json.loads(p.read_text(encoding="utf-8")))
    return cfg


def deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_credentials(camera: dict) -> tuple[str, str]:
    """Las credenciales vienen del entorno. Nunca del JSON."""
    user_env = camera.get("username_env", "CAMERA_USER")
    pass_env = camera.get("password_env", "CAMERA_PASS")
    user = os.environ.get(user_env, "")
    password = os.environ.get(pass_env, "")
    if not user or not password:
        raise SystemExit(
            f"Faltan credenciales. Define las variables {user_env} y {pass_env}."
        )
    return user, password


# ----------------------------- Captura ------------------------------

def grab_isapi_snapshot(cfg: dict) -> np.ndarray | None:
    """Descarga un fotograma por HTTP (ISAPI de Hikvision y compatibles)."""
    url = cfg["camera"].get("isapi_snapshot_url")
    if not url:
        return None
    user, password = resolve_credentials(cfg["camera"])

    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    base = re.match(r"(https?://[^/]+)", url)
    if base:
        mgr.add_password(None, base.group(1), user, password)
    opener = urllib.request.build_opener(urllib.request.HTTPDigestAuthHandler(mgr))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smoke-detect"})
        with opener.open(req, timeout=cfg["camera"]["timeout"]) as resp:
            data = resp.read()
        if len(data) < 1000:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"  ! snapshot HTTP fallo: {exc}", file=sys.stderr)
        return None


def inject_credentials(url: str, user: str, password: str) -> str:
    """
    Inserta usuario:contrasena en la URL RTSP sin dejarlos en disco.

    Se codifican con quote() para que una contrasena con @ : / # no rompa
    la URL (las de las camaras Hikvision suelen llevar simbolos).
    """
    host_part = url.split("://", 1)[-1].split("/", 1)[0]
    if "@" in host_part:
        return url
    scheme, rest = url.split("://", 1)
    u = urllib.parse.quote(user, safe="")
    p = urllib.parse.quote(password, safe="")
    return f"{scheme}://{u}:{p}@{rest}"


def grab_rtsp_frame(cfg: dict) -> np.ndarray | None:
    """Toma un fotograma de un stream RTSP con OpenCV."""
    url = cfg["camera"].get("rtsp_url")
    if not url:
        return None
    user, password = resolve_credentials(cfg["camera"])
    safe_url = inject_credentials(url, user, password)

    cap = cv2.VideoCapture(safe_url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = False, None
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.3)
        return frame if ok else None
    finally:
        cap.release()


def grab_burst(cfg: dict) -> list[np.ndarray]:
    """Captura N fotogramas separados en el tiempo para medir movimiento."""
    n = int(cfg["camera"].get("burst_frames", 5))
    gap = float(cfg["camera"].get("burst_interval", 0.9))
    frames: list[np.ndarray] = []

    for i in range(n):
        frame = grab_isapi_snapshot(cfg)
        if frame is None:
            break
        frames.append(frame)
        if i < n - 1:
            time.sleep(gap)

    if len(frames) >= 2:
        return frames

    frame = grab_rtsp_frame(cfg)
    if frame is not None:
        return [frame]
    # Si RTSP tambien fallo, al menos usar el unico fotograma ISAPI.
    return frames


# ----------------------------- Analisis -----------------------------

def scale_box(box: list[int], width: int, height: int, ref_size: list[int]) -> tuple:
    """Reescala un rectangulo definido sobre la resolucion de referencia."""
    rx, ry = ref_size
    sx, sy = width / rx, height / ry
    x1, y1, x2, y2 = box
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def crop(img: np.ndarray, box: tuple) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = img[y1:y2, x1:x2]
    return region if region.size else None


def zone_metrics(img: np.ndarray, box: tuple) -> dict | None:
    """Luminancia media y saturacion media de una zona."""
    region = crop(img, box)
    if region is None:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return {
        "luma": float(np.mean(gray)),
        "sat": float(np.mean(hsv[:, :, 1])),
        "region": region,
    }


def relative_gain(value_zone: float, value_ref: float,
                  higher_means_better: bool = True) -> float:
    """
    Diferencia relativa en porcentaje entre zona y referencia.

    El humo sube la luminancia (dispersa luz) y baja la saturacion (blanquea).
    La referencia va en el denominador para que la medida sea independiente
    del nivel absoluto de luz de la escena.
    """
    denom = max(value_ref, 1.0)
    if higher_means_better:
        return max(0.0, (value_zone - value_ref) / denom) * 100.0
    return max(0.0, (value_ref - value_zone) / denom) * 100.0


def is_night(img: np.ndarray, cfg: dict) -> bool:
    """Filtro nocturno: de noche el analisis de contraste no es fiable."""
    if not cfg["night"].get("enabled", True):
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < float(cfg["night"].get("mean_luma_below", 85.0))


def classify(opacity: float, cfg: dict) -> tuple[str, str, int]:
    """Devuelve (codigo, etiqueta, escala Ringelmann)."""
    c = cfg["classification"]
    clear = float(c.get("clear_below", 5.0))
    gray_from = float(c.get("gray_from", 40.0))
    dark_from = float(c.get("dark_from", 70.0))

    if opacity < clear:
        code, label = "CLEAR", "Despejado / Sin humo"
    elif opacity < gray_from:
        code, label = "LIGHT", "Humo claro / blanco"
    elif opacity < dark_from:
        code, label = "GRAY", "Humo gris"
    else:
        code, label = "DARK", "Humo oscuro / combustion incompleta"

    return code, label, ringelmann_number(opacity)


def ringelmann_number(opacity: float) -> int:
    """
    Numero Ringelmann por la escala estandar: cada grado equivale a 20%
    de opacidad (1 = 20%, 2 = 40%, 3 = 60%, 4 = 80%, 5 = 100%).
    Se redondea al grado mas cercano, como se hace en la lectura visual.
    """
    return int(min(5, max(0, round(opacity / 20.0))))


def raw_signal(img: np.ndarray, cfg: dict) -> float | None:
    """Calcula la senal cruda (sin calibrar) de una imagen."""
    h, w = img.shape[:2]
    ref_size = cfg["zones"]["reference_size"]
    plume_box = scale_box(cfg["zones"]["plume"], w, h, ref_size)
    ref_a_box = scale_box(cfg["zones"]["reference_a"], w, h, ref_size)
    ref_b_box = scale_box(cfg["zones"]["reference_b"], w, h, ref_size)

    plume = zone_metrics(img, plume_box)
    ref_a = zone_metrics(img, ref_a_box)
    ref_b = zone_metrics(img, ref_b_box)
    if not (plume and ref_a and ref_b):
        return None

    weights = cfg["calibration"]["weights"]
    return (
        relative_gain(plume["luma"], ref_a["luma"], True) * weights["luma_gain"]
        + relative_gain(plume["sat"], ref_a["sat"], False) * weights["sat_loss_ref_a"]
        + relative_gain(plume["sat"], ref_b["sat"], False) * weights["sat_loss_ref_b"]
    )


def analyze(frames: list[np.ndarray], cfg: dict) -> dict | None:
    """
    Analiza una rafaga (o una sola imagen) y devuelve opacidad y clasificacion.

    Calibracion lineal:  opacidad = slope * senal + offset
    Los dos parametros salen de dos muestras conocidas (ver --calibrate).
    """
    frames = [f for f in frames if f is not None]
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    ref_size = cfg["zones"]["reference_size"]
    plume_box = scale_box(cfg["zones"]["plume"], w, h, ref_size)
    ref_a_box = scale_box(cfg["zones"]["reference_a"], w, h, ref_size)
    ref_b_box = scale_box(cfg["zones"]["reference_b"], w, h, ref_size)

    if is_night(frames[-1], cfg):
        # De noche NO se reporta "despejado": no se sabe. Un sistema de
        # alertas debe poder distinguir "sin humo" de "sin medicion".
        return {
            "code": "NIGHT",
            "label": "Sin medicion (modo nocturno)",
            "opacity": None,
            "ringelmann": None,
            "night": True,
            "frames_used": len(frames),
            "annotated": frames[-1],
            "plume_crop": crop(frames[-1], plume_box),
        }

    signals = []
    for frame in frames:
        s = raw_signal(frame, cfg)
        if s is None:
            continue
        plume = zone_metrics(frame, plume_box)
        ref_a = zone_metrics(frame, ref_a_box)
        signals.append({
            "raw": s,
            "luma": plume["luma"],
            "luma_ref": ref_a["luma"],
            "sat": plume["sat"],
        })

    if not signals:
        return None

    mean_signal = float(np.mean([s["raw"] for s in signals]))
    slope = float(cfg["calibration"]["slope"])
    offset = float(cfg["calibration"]["offset"])
    opacity = round(min(100.0, max(0.0, slope * mean_signal + offset)), 1)

    motion = 0.0
    if len(frames) >= 2:
        diffs = []
        for a, b in zip(frames, frames[1:]):
            ca, cb = crop(a, plume_box), crop(b, plume_box)
            if ca is None or cb is None or ca.shape != cb.shape:
                continue
            ga = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gb = cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diffs.append(float(np.mean(np.abs(ga - gb))))
        motion = round(float(np.mean(diffs)), 2) if diffs else 0.0

    code, label, ringelmann = classify(opacity, cfg)
    last = signals[-1]

    return {
        "code": code,
        "label": label,
        "opacity": opacity,
        "ringelmann": ringelmann,
        "night": False,
        "frames_used": len(signals),
        "motion": motion,
        "luma": round(last["luma"], 1),
        "luma_ref": round(last["luma_ref"], 1),
        "sat": round(last["sat"], 1),
        "annotated": annotate(frames[-1], plume_box, ref_a_box, ref_b_box,
                              label, opacity, ringelmann),
        "plume_crop": crop(frames[-1], plume_box),
    }


def annotate(img: np.ndarray, plume: tuple, ref_a: tuple, ref_b: tuple,
             label: str, opacity: float, ringelmann: int) -> np.ndarray:
    """Dibuja las tres zonas y los valores medidos sobre la imagen."""
    out = img.copy()
    green = (0, 255, 0)
    for box in (plume, ref_a, ref_b):
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), green, 3)

    x1, y1 = plume[0], plume[1]
    cv2.putText(out, label.upper(), (max(10, x1 - 100), max(30, y1 - 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2)
    cv2.putText(out, f"Opacidad {opacity:.1f}% | Ringelmann {ringelmann}",
                (max(10, x1 - 100), y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


# --------------------------- Calibracion ----------------------------

def calibrate(clear_img: np.ndarray, smoke_img: np.ndarray, cfg: dict,
              known_opacity: float = 30.0) -> dict:
    """
    Deriva slope y offset a partir de dos imagenes: una sin humo y una con
    humo de opacidad conocida (por defecto 30%, referencia tipica).

    Dos incognitas, dos ecuaciones. Se mide la senal cruda en ambas y se
    resuelve el sistema lineal.
    """
    smoke_signal = raw_signal(smoke_img, cfg)
    if smoke_signal is None:
        raise SystemExit("No se pudieron medir las zonas en la imagen con humo.")

    clear_signal = raw_signal(clear_img, cfg) or 0.0

    if smoke_signal <= clear_signal:
        raise SystemExit(
            "La imagen con humo no muestra mas senal que la limpia. Revisa las zonas."
        )

    slope = (known_opacity - 0.0) / (smoke_signal - clear_signal)
    offset = -slope * clear_signal

    return {
        "slope": round(slope, 6),
        "offset": round(offset, 6),
        "clear_signal": round(clear_signal, 2),
        "smoke_signal": round(smoke_signal, 2),
        "known_opacity": known_opacity,
    }


# ----------------------------- Salida -------------------------------

def save_results(result: dict, cfg: dict, prefix: str = "smoke") -> dict:
    """Guarda las imagenes y devuelve las rutas."""
    out_dir = Path(cfg["output"].get("dir", "captures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = {}

    if cfg["output"].get("save_annotated", True) and result.get("annotated") is not None:
        path = out_dir / f"{prefix}_{stamp}_annotated.jpg"
        cv2.imwrite(str(path), result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 82])
        saved["annotated"] = str(path)

    if result.get("plume_crop") is not None:
        path = out_dir / f"{prefix}_{stamp}_plume.jpg"
        cv2.imwrite(str(path), result["plume_crop"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved["plume"] = str(path)

    return saved


def print_result(result: dict, saved: dict | None = None) -> None:
    print(f"  Estado      : {result['label']}  [{result['code']}]")
    if result.get("opacity") is not None:
        print(f"  Opacidad    : {result['opacity']:.1f}%")
        print(f"  Ringelmann  : {result['ringelmann']}")
    if result.get("night"):
        print("  Nota        : imagen nocturna, analisis de contraste no aplicable")
    else:
        print(f"  Luminancia  : {result.get('luma')} (ref {result.get('luma_ref')})")
        print(f"  Movimiento  : {result.get('motion', 0)}")
        print(f"  Fotogramas  : {result.get('frames_used')}")
    for key, path in (saved or {}).items():
        print(f"  {key:<12}: {path}")


# ------------------------------- CLI --------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detector de humo por contraste relativo en camaras IP."
    )
    ap.add_argument("--config", help="JSON de configuracion (camara, zonas, calibracion)")
    ap.add_argument("--snapshot", action="store_true",
                    help="Guardar un fotograma para elegir zonas")
    ap.add_argument("--measure", action="store_true",
                    help="Capturar rafaga y medir opacidad")
    ap.add_argument("--image", help="Analizar un archivo de imagen local")
    ap.add_argument("--calibrate", nargs=2, metavar=("SIN_HUMO", "CON_HUMO"),
                    help="Derivar slope y offset de dos imagenes de referencia")
    ap.add_argument("--known-opacity", type=float, default=30.0,
                    help="Opacidad real de la muestra con humo, en %% (default 30)")
    ap.add_argument("--json", action="store_true", help="Salida en JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.calibrate:
        clear = cv2.imread(args.calibrate[0])
        smoke = cv2.imread(args.calibrate[1])
        if clear is None or smoke is None:
            print("No se pudieron leer las imagenes.", file=sys.stderr)
            return 1
        cal = calibrate(clear, smoke, cfg, args.known_opacity)
        print("--- Calibracion derivada ---")
        print(json.dumps(cal, indent=2, ensure_ascii=False))
        print("")
        print("Copia estos valores en la seccion 'calibration' de tu config:")
        print(f'  "slope": {cal["slope"]},')
        print(f'  "offset": {cal["offset"]}')
        return 0

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"No se pudo leer la imagen: {args.image}", file=sys.stderr)
            return 1
        result = analyze([frame], cfg)
        if not result:
            print("No se pudo analizar la imagen.", file=sys.stderr)
            return 1
        saved = save_results(result, cfg, prefix="image")
        if args.json:
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("annotated", "plume_crop")},
                             indent=2, ensure_ascii=False))
        else:
            print_result(result, saved)
        return 0

    if args.snapshot:
        # No usar `a or b`: un ndarray no tiene valor de verdad y truena.
        frame = grab_isapi_snapshot(cfg)
        if frame is None:
            frame = grab_rtsp_frame(cfg)
        if frame is None:
            print("No se pudo capturar un fotograma.", file=sys.stderr)
            return 1
        out_dir = Path(cfg["output"].get("dir", "captures"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(path), frame)
        h, w = frame.shape[:2]
        print(f"Fotograma guardado: {path}  ({w}x{h})")
        print(f"Ajusta 'zones.reference_size' a [{w}, {h}] si cambiaste de resolucion.")
        return 0

    if args.measure:
        frames = grab_burst(cfg)
        if not frames:
            print("No se pudo capturar la rafaga.", file=sys.stderr)
            return 1
        result = analyze(frames, cfg)
        if not result:
            print("El analisis no devolvio resultado. Revisa las zonas.", file=sys.stderr)
            return 1
        saved = save_results(result, cfg)
        if args.json:
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("annotated", "plume_crop")},
                             indent=2, ensure_ascii=False))
        else:
            print(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} ---")
            print_result(result, saved)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
