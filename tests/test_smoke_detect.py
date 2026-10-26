"""Pruebas sin camara: se generan escenas sinteticas con numpy."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke_detect as sd  # noqa: E402

SIZE = (1520, 2688)  # alto, ancho = reference_size por defecto


def scene(plume_bgr=(90, 120, 150), bg_bgr=(90, 120, 150)):
    """Escena plana color `bg`, con la zona de pluma pintada de `plume`."""
    img = np.zeros((*SIZE, 3), np.uint8)
    img[:] = bg_bgr
    x1, y1, x2, y2 = sd.DEFAULT_CONFIG["zones"]["plume"]
    img[y1:y2, x1:x2] = plume_bgr
    return img


@pytest.fixture
def cfg():
    return sd.load_config(None)


def test_ringelmann_standard_scale():
    assert sd.ringelmann_number(0) == 0
    assert sd.ringelmann_number(9) == 0
    assert sd.ringelmann_number(20) == 1
    assert sd.ringelmann_number(40) == 2
    assert sd.ringelmann_number(60) == 3
    assert sd.ringelmann_number(80) == 4
    assert sd.ringelmann_number(100) == 5


def test_clear_scene_is_clear(cfg):
    res = sd.analyze([scene()], cfg)
    assert res["code"] == "CLEAR"
    assert res["night"] is False


def test_smoke_scene_detected(cfg):
    # Pluma mas clara y desaturada que el fondo.
    res = sd.analyze([scene(plume_bgr=(215, 215, 220))], cfg)
    assert res["code"] in ("LIGHT", "GRAY", "DARK")
    assert res["opacity"] > cfg["classification"]["clear_below"]


def test_night_is_not_reported_as_clear(cfg):
    res = sd.analyze([scene(plume_bgr=(10, 10, 10), bg_bgr=(10, 10, 10))], cfg)
    assert res["code"] == "NIGHT"
    assert res["opacity"] is None
    assert res["ringelmann"] is None


def test_calibration_hits_both_points(cfg):
    clear, smoke = scene(), scene(plume_bgr=(215, 215, 220))
    cal = sd.calibrate(clear, smoke, cfg, known_opacity=30)
    cfg["calibration"]["slope"] = cal["slope"]
    cfg["calibration"]["offset"] = cal["offset"]
    assert sd.analyze([clear], cfg)["opacity"] == pytest.approx(0, abs=0.2)
    assert sd.analyze([smoke], cfg)["opacity"] == pytest.approx(30, abs=0.2)


def test_credentials_are_url_encoded():
    url = sd.inject_credentials("rtsp://cam:554/ch1", "admin", "p@ss:w/rd#1")
    assert url == "rtsp://admin:p%40ss%3Aw%2Frd%231@cam:554/ch1"


def test_existing_credentials_untouched():
    url = "rtsp://u:p@cam/ch1"
    assert sd.inject_credentials(url, "x", "y") == url


def test_snapshot_fallback_does_not_crash(cfg, monkeypatch, tmp_path, capsys):
    """Regresion: `snapshot or rtsp` truena con ndarray."""
    monkeypatch.setattr(sd, "grab_isapi_snapshot", lambda c: scene())
    monkeypatch.setattr(sd, "grab_rtsp_frame", lambda c: None)
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text('{"output": {"dir": "%s"}}' % tmp_path.as_posix())
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path), "--snapshot"])
    assert sd.main() == 0
    assert "Fotograma guardado" in capsys.readouterr().out
