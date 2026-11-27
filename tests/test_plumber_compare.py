"""FLUXlight against AIA (A4-03): the compare route and the MCP tool, on the fake backend's synthetic FITS pair.

The fake ``euv`` scenario writes a model image and a stand-in AIA image of the same scene on one helioprojective
header; the AIA image is the model times 2, so the summary numbers have known values."""

import base64
import glob
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")
pytest.importorskip("sunpy")

sys.path.insert(0, os.path.dirname(__file__))
from test_plumber_mcp import H, Rig, call_raw, done, rig  # noqa: E402,F401

from fluxpype.plumber import compare  # noqa: E402


@pytest.fixture
def euv_task(rig):
    view = rig.run_job(scenario="euv", speed=300)
    assert view["status"] == "ok", view
    return view["tasks"][0]["id"]


def test_euv_is_a_stage_and_the_fake_run_reports_it(rig, euv_task):
    t = rig.tc.get(f"/api/tasks/{euv_task}", headers=H).json()
    assert [s["id"] for s in t["stages"]][-2:] == ["fluxlight", "euv"]
    assert next(s for s in t["stages"] if s["id"] == "euv")["status"] == "ok"
    pair = [a for a in t["artifacts"] if a["kind"] == "fits"]
    assert len(pair) == 1 and pair[0]["pair"].endswith(".fits")


def test_compare_route_returns_aligned_arrays_and_known_numbers(rig, euv_task):
    r = rig.tc.get(f"/api/tasks/{euv_task}/compare", headers=H).json()
    assert r["available"] and r["channel"] == "193" and r["channels"] == ["193"]
    assert r["shape"] == [64, 64] and r["unit"] == "DN s-1 pix-1"
    model, aia = compare.unpack(r["model"], r["shape"]), compare.unpack(r["aia"], r["shape"])
    assert model.shape == aia.shape == (64, 64)
    ok = np.isfinite(model) & np.isfinite(aia) & (model > 0) & (aia > 0)
    assert np.nanmedian(aia[ok] / model[ok]) == pytest.approx(2.0, rel=0.05)  # AIA is the model times 2
    m = r["metrics"]
    assert m["model_over_aia_total"] == pytest.approx(0.5, rel=0.05)
    assert m["median_log_ratio"] == pytest.approx(np.log10(0.5), abs=0.03)
    assert m["log_correlation"] > 0.99 and m["pixels_compared"] > 1000
    assert r["wcs"]["CRPIX1"] == pytest.approx(32.5) and r["wcs"]["RSUN_OBS"] == 960.0
    assert r["stretch"][0] < r["stretch"][1]


def test_compare_route_display_size_scales_the_wcs(rig, euv_task):
    full = rig.tc.get(f"/api/tasks/{euv_task}/compare", headers=H).json()
    small = rig.tc.get(f"/api/tasks/{euv_task}/compare?size=32&arrays=false", headers=H).json()
    assert small["shape"] == [32, 32] and small["binning"] == 2 and "data" not in small["model"]
    assert small["wcs"]["CDELT1"] == pytest.approx(2 * full["wcs"]["CDELT1"])
    assert small["wcs"]["CRPIX1"] == pytest.approx(16.5)  # the Sun's centre stays at the centre
    assert small["metrics"] == full["metrics"]  # the numbers come from the full-size pair


def test_compare_route_says_so_when_the_aia_file_is_missing(rig, euv_task):
    for f in glob.glob(os.path.join(rig.sched.data_dir(), "aia", "*.fits")):
        os.remove(f)
    r = rig.tc.get(f"/api/tasks/{euv_task}/compare", headers=H).json()
    assert r["available"] and r["aia_missing"] and r["aia"] is None and r["metrics"] is None
    assert "not on this machine" in r["note"] and "fluxpype.science.aia fetch" in r["note"]
    assert compare.unpack(r["model"], r["shape"]).shape == (64, 64)  # the model image is still there


def test_compare_route_errors_and_empty_tasks(rig, done, euv_task):
    assert rig.tc.get("/api/tasks/nope/compare", headers=H).status_code == 404
    assert rig.tc.get(f"/api/tasks/{euv_task}/compare?channel=x", headers=H).status_code == 400
    r = rig.tc.get(f"/api/tasks/{euv_task}/compare?channel=171", headers=H).json()
    assert not r["available"] and "193" in r["note"]
    r = rig.tc.get(f"/api/tasks/{done[1]}/compare", headers=H).json()  # a plain run has no EUV image
    assert not r["available"] and r["channels"] == [] and "no FLUXlight EUV image" in r["note"]


def test_a_pair_outside_the_data_folder_is_not_followed(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.fits").write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    model = data / "m_aia193.fits"
    model.write_bytes(b"")
    arts = [{"path": str(model), "kind": "fits", "pair": str(outside / "x.fits")}]
    assert compare.find_aia_file(str(model), "193", arts, str(data)) is None


def test_mcp_get_comparison_returns_the_numbers_and_a_picture(rig, euv_task):
    res = call_raw(rig, "get_comparison", task_id=euv_task, max_edge=64)
    assert not res.is_error
    assert [c.type for c in res.content] == ["text", "image"] and res.content[1].mime_type == "image/png"
    body = json.loads(res.content[0].text)
    route = rig.tc.get(f"/api/tasks/{euv_task}/compare?size=64", headers=H).json()
    assert body["metrics"] == route["metrics"] and body["wcs"] == route["wcs"] and "data" not in body["model"]
    png = base64.b64decode(res.content[1].data)
    assert png[:4] == b"\x89PNG"
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(png)).size == (3 * 64, 64)  # FLUXlight | AIA | log ratio


def test_mcp_get_comparison_without_aia_sends_the_model_only_and_says_so(rig, euv_task):
    for f in glob.glob(os.path.join(rig.sched.data_dir(), "aia", "*.fits")):
        os.remove(f)
    res = call_raw(rig, "get_comparison", task_id=euv_task, max_edge=64)
    body = json.loads(res.content[0].text)
    assert body["aia_missing"] and "not on this machine" in body["note"] and len(res.content) == 2
