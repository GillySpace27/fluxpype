"""Cache index and trash: classification, and that nothing can escape data_dir."""

import json
import os

import pytest

from fluxpype.studio.cache import Cache, CacheError, classify


def make(tmp_path):
    data = tmp_path / "data"
    files = {
        "magnetograms/CR2225_r4_hmi.fits": 1000,
        "magnetograms/CR2225_rf2_adapt.fits": 900,
        "pfss/pfss_cr2225_r4_hmi.pkl": 5000,
        "batches/b1/data/cr2225/floc/floc_cr2225_r4_f100_hmi.dat": 100,
        "batches/b1/data/cr2225/floc/floc_cr2225_rf2_f100_adapt.dat": 100,
        "batches/b1/data/cr2225/world/cr2225_f100_hmi.flux": 2000,
        "batches/b1/data/cr2225/world/cr2225_f100_hmi_relaxed_s800.flux": 2100,
        "batches/b1/data/cr2225/wind/cr2225_f100_parker_wind.dat": 300,
        "batches/b1/data/cr2225/wind/wind.png": 50,
        "batches/b1/data/cr2225/light/cr2225_fluxlight_light_parker_1.png": 70,
        "batches/b1/data/cr2225/world/flux-movie-00001.png": 20,
        "notes.txt": 3,
    }
    for rel, size in files.items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    return data


def test_classification():
    cases = {
        "magnetograms/CR2225_r4_hmi.fits": ("magnetogram", {"cr": "2225", "reduce": "4", "inst": "hmi"}),
        "magnetograms/CR2225_rf2_adapt.fits": ("magnetogram", {"cr": "2225", "sel": "2", "inst": "adapt"}),
        "pfss/pfss_cr2225_r4_hmi.pkl": ("pfss_pickle", {"cr": "2225", "inst": "hmi"}),
        "batches/b/data/cr2225/floc/floc_cr2225_r4_f100_hmi.dat": ("footpoints", {"cr": "2225", "n": "100", "batch": "b"}),
        "batches/b/data/cr2225/world/cr2225_f100_adapt.flux": ("world", {"n": "100", "inst": "adapt"}),
        "batches/b/data/cr2225/world/cr2225_f100_adapt_relaxed_s900.flux": ("relaxed_world", {"steps": "900", "inst": "adapt"}),
        "batches/b/data/cr2225/wind/x.dat": ("wind", {"batch": "b", "cr": "2225"}),
        "batches/b/data/cr2225/wind/x.png": ("plots", {}),
        "batches/b/data/cr2225/light/cr2225_fluxlight_a.png": ("fluxlight", {"cr": "2225"}),
        "misc/readme.txt": ("other", {}),
        ".flux_trash/20260101/x": ("trash", {}),
    }
    for rel, (kind, extra) in cases.items():
        info = classify(rel)
        assert info["kind"] == kind, rel
        for k, v in extra.items():
            assert info[k] == v, (rel, k)


def test_scan_and_summary(tmp_path):
    data = make(tmp_path)
    c = Cache(str(data))
    s = c.summary(rescan=True)
    tiles = {t["kind"]: t for t in s["tiles"]}
    assert s["total_count"] == 12 and tiles["magnetogram"]["count"] == 2 and tiles["magnetogram"]["size"] == 1900
    assert tiles["relaxed_world"]["count"] == 1 and tiles["fluxlight"]["count"] == 1
    assert tiles["wind"]["count"] == 1 and tiles["plots"]["count"] == 2
    assert s["trash"] == {"count": 0, "size": 0}


def test_missing_data_dir_is_an_empty_cache(tmp_path):
    s = Cache(str(tmp_path / "nope")).summary()
    assert s["exists"] is False and s["total_count"] == 0


def test_trash_roundtrip(tmp_path):
    data = make(tmp_path)
    c = Cache(str(data))
    res = c.move_to_trash(["pfss/pfss_cr2225_r4_hmi.pkl", "notes.txt"])
    assert res["size"] == 5003
    assert not (data / "pfss/pfss_cr2225_r4_hmi.pkl").exists()
    folder = data / ".flux_trash" / res["folder"]
    assert (folder / "pfss/pfss_cr2225_r4_hmi.pkl").exists()
    assert json.loads((folder / "manifest.json").read_text())["files"] == ["pfss/pfss_cr2225_r4_hmi.pkl", "notes.txt"]
    assert c.summary(rescan=True)["total_count"] == 10  # trash is not part of the index
    items = c.list_trash()
    assert {i["path"] for i in items} == {"pfss/pfss_cr2225_r4_hmi.pkl", "notes.txt"}
    pfss = next(i for i in items if i["path"].startswith("pfss"))
    assert c.restore([pfss["id"]]) == [pfss["path"]]
    assert (data / "pfss/pfss_cr2225_r4_hmi.pkl").exists()
    assert c.summary(rescan=True)["trash"]["count"] == 1
    (data / "notes.txt").write_text("new")  # restoring over an existing file never overwrites
    with pytest.raises(CacheError):
        c.restore([i["id"] for i in c.list_trash() if i["path"] == "notes.txt"])
    assert (data / "notes.txt").read_text() == "new"
    out = c.empty_trash()
    assert out["removed"] == 1
    assert c.list_trash() == []


@pytest.mark.parametrize("bad", ["../outside.txt", "a/../../outside.txt", "/etc/passwd", "", ".flux_trash/x", "pfss/../../outside.txt"])
def test_trash_refuses_paths_outside_data_dir(tmp_path, bad):
    data = make(tmp_path)
    (tmp_path / "outside.txt").write_text("keep")
    c = Cache(str(data))
    with pytest.raises(CacheError):
        c.move_to_trash([bad])
    assert (tmp_path / "outside.txt").read_text() == "keep"


def test_trash_refuses_symlink_escapes(tmp_path):
    data = make(tmp_path)
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "file.txt").write_text("keep")
    os.symlink(secret, data / "linkdir")  # a directory link leading out
    os.symlink(secret / "file.txt", data / "linkfile")  # a file link leading out
    c = Cache(str(data))
    for bad in ("linkdir/file.txt", "linkfile"):
        with pytest.raises(CacheError):
            c.move_to_trash([bad])
    assert (secret / "file.txt").read_text() == "keep"
    # the scan does not follow links either
    assert "linkdir/file.txt" not in {r["path"] for r in c.records(rescan=True)}


def test_one_bad_path_means_nothing_moves(tmp_path):
    data = make(tmp_path)
    c = Cache(str(data))
    with pytest.raises(CacheError):
        c.move_to_trash(["notes.txt", "../x"])
    assert (data / "notes.txt").exists()
    with pytest.raises(CacheError):
        c.move_to_trash(["does/not/exist"])


def test_restore_rejects_traversal(tmp_path):
    data = make(tmp_path)
    c = Cache(str(data))
    c.move_to_trash(["notes.txt"])
    for bad in ("../notes.txt", "x/../../etc/passwd", "nonsense", "20260101/../../x"):
        with pytest.raises(CacheError):
            c.restore([bad])


def test_empty_trash_only_touches_the_trash(tmp_path):
    data = make(tmp_path)
    c = Cache(str(data))
    c.move_to_trash(["notes.txt"])
    c.empty_trash()
    assert (data / "pfss/pfss_cr2225_r4_hmi.pkl").exists()
    assert not (data / "notes.txt").exists()
