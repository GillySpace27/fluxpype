"""Cache viewer: file previews, FITS headers without astropy, rendering through the job Python, path safety."""

import gzip
import os
import shutil
import struct

import pytest

from fluxpype.plumber import preview


def fits_bytes(nx=4, ny=3, extra_cards=()):
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    2",
        f"NAXIS1  = {nx:20d}",
        f"NAXIS2  = {ny:20d}",
        "TELESCOP= 'SDO/HMI'            / the instrument",
        "COMMENT  synoptic map",
        *extra_cards,
        "END",
    ]
    head = "".join(c.ljust(80) for c in cards).encode("ascii")
    head += b" " * (-len(head) % 2880)
    data = b"".join(struct.pack(">f", float(i - 5)) for i in range(nx * ny))
    data += b"\0" * (-len(data) % 2880)
    return head + data


def test_fits_header_is_read_without_astropy(tmp_path):
    p = tmp_path / "CR2225_r4_hmi.fits"
    p.write_bytes(fits_bytes())
    hdus = preview.fits_headers(str(p))
    assert len(hdus) == 1 and hdus[0]["dims"] == [4, 3]
    cards = {c["key"]: c for c in hdus[0]["cards"]}
    assert cards["TELESCOP"]["value"] == "SDO/HMI" and cards["TELESCOP"]["comment"] == "the instrument"
    d = preview.describe(str(p), p.name)
    assert d["view"] == "fits" and d["renderable"]


def test_gzipped_fits_and_a_second_hdu(tmp_path):
    ext = "".join(
        c.ljust(80)
        for c in [
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTNAME = 'MASK'",
            "END",
        ]
    ).encode()
    ext += b" " * (-len(ext) % 2880)
    p = tmp_path / "x.fits.gz"
    p.write_bytes(gzip.compress(fits_bytes() + ext))
    hdus = preview.fits_headers(str(p))
    assert [h["name"] for h in hdus] == ["PRIMARY", "MASK"]


def test_text_binary_and_pickle(tmp_path):
    t = tmp_path / "floc_cr2225_r4_f100_hmi.dat"
    t.write_text("\n".join(f"{i} 0.1 0.2" for i in range(1000)))
    d = preview.describe(str(t), t.name)
    assert d["view"] == "text" and len(d["lines"]) == preview.TEXT_LINES and d["truncated"]
    b = tmp_path / "blob.bin"
    b.write_bytes(bytes(range(256)) * 10)
    assert preview.describe(str(b), b.name)["view"] == "none"
    k = tmp_path / "pfss_cr2225_r4_hmi.pkl"
    k.write_bytes(b"\x80\x04not really")
    d = preview.describe(str(k), k.name)
    assert d["view"] == "none" and "pickle" in d["note"]
    bad = tmp_path / "broken.fits"
    bad.write_bytes(b"hello")
    assert preview.describe(str(bad), bad.name)["view"] == "none"


def _python_with_astropy():
    for cand in (os.environ.get("FLUX_TEST_ASTROPY_PYTHON"), shutil.which("python3")):
        if cand and os.access(cand, os.X_OK):
            import subprocess

            if subprocess.run([cand, "-c", "import numpy, astropy, matplotlib"], capture_output=True).returncode == 0:
                return cand
    return None


def test_render_fits_with_the_job_python(tmp_path):
    py = _python_with_astropy()
    if not py:
        pytest.skip("no interpreter with numpy, astropy and matplotlib")
    p = tmp_path / "m.fits"
    p.write_bytes(fits_bytes(40, 20))
    out = preview.render_fits(str(p), str(tmp_path / "home"), py)
    assert os.path.getsize(out["png"]) > 100 and out["stats"]["shape"] == [20, 40]
    again = preview.render_fits(str(p), str(tmp_path / "home"), py)  # cached
    assert again["png"] == out["png"]


def test_render_reports_a_missing_science_stack(tmp_path):
    p = tmp_path / "m.fits"
    p.write_bytes(fits_bytes())
    with pytest.raises(preview.RenderError):
        preview.render_fits(str(p), str(tmp_path / "home"), str(tmp_path / "no-python"))


def test_api_preview_and_file_are_confined(make_scheduler, plumber_config, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fluxpype.plumber.backends.fake import FakeBackend
    from fluxpype.plumber.server import create_app

    b = FakeBackend()
    b.kind = "local"
    sched = make_scheduler(backends={"local": b})
    data = tmp_path / "data"
    (data / "magnetograms").mkdir(parents=True)
    (data / "magnetograms" / "CR2225_r4_hmi.fits").write_bytes(fits_bytes())
    (data / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "secret.txt").write_text("no")
    app = create_app(sched, "tok", demo=False, tick_interval=0.1)
    h = {"X-Flux-Token": "tok"}
    with TestClient(app, base_url="http://localhost") as c:
        r = c.get("/api/cache/preview", params={"path": "magnetograms/CR2225_r4_hmi.fits"}, headers=h)
        assert r.status_code == 200 and r.json()["view"] == "fits"
        r = c.get("/api/cache/file", params={"path": "plot.png"}, headers=h)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.headers["content-disposition"].startswith("inline")
        r = c.get("/api/cache/file", params={"path": "plot.png", "download": 1}, headers=h)
        assert r.headers["content-disposition"].startswith("attachment")
        for bad in ("../secret.txt", "/etc/passwd", "magnetograms/../../secret.txt"):
            assert c.get("/api/cache/preview", params={"path": bad}, headers=h).status_code == 403
            assert c.get("/api/cache/file", params={"path": bad}, headers=h).status_code == 403
        assert c.get("/api/cache/render", params={"path": "plot.png"}, headers=h).status_code == 415
        assert c.get("/api/cache/file", params={"path": "plot.png"}).status_code == 401
