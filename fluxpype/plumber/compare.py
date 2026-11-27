"""FLUXlight against AIA for one task and channel: the data behind the compare page and the MCP tool (A4-03).

Nothing here computes an image. The model image is the per-channel FITS the pipeline wrote (science/light_fits.py);
the AIA image is the prepped level 1.5 FITS science/aia.py fetched; ``aia.compare`` reprojects the AIA image onto the
model grid and gives the summary numbers. This module only finds the two files, shrinks both arrays to a display size
(block mean, WCS scaled to match) and packs them as base64 float32 so the browser can change the stretch without a
round trip. When the AIA file is not on disk the answer says so and carries the model image alone.
"""

import base64
import glob
import os
import re
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

MODEL_NAME = re.compile(r"_aia(\d+)\.fits$")
SIZE_MIN, SIZE_MAX, SIZE_DEFAULT = 16, 2048, 512
WCS_KEYS = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "RSUN_OBS", "DSUN_OBS", "DATE-OBS")
FETCH_HINT = "python -m fluxpype.science.aia fetch --cr {cr} --channel {ch} --out <data_dir>/aia"

_cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_lock = threading.Lock()


def clamp_size(size) -> int:
    try:
        return max(SIZE_MIN, min(SIZE_MAX, int(size)))
    except (TypeError, ValueError):
        return SIZE_DEFAULT


def _under(path: str, root: str) -> bool:
    real, base = os.path.realpath(path), os.path.realpath(root)
    return real == base or real.startswith(base + os.sep)


def find_model_files(products: List[Dict[str, Any]]) -> Dict[str, str]:
    """Channel -> data-dir-relative path of the newest model FITS (``..._aia<ch>.fits``) among the FLUXlight files."""
    best: Dict[str, Dict[str, Any]] = {}
    for prod in products:
        if prod.get("id") != "fluxlight":
            continue
        for f in prod.get("files", []):
            m = MODEL_NAME.search(f.get("name", ""))
            if m and (m.group(1) not in best or f.get("mtime", 0) > best[m.group(1)].get("mtime", 0)):
                best[m.group(1)] = f
    return {ch: f["path"] for ch, f in best.items()}


def find_aia_file(
    model_abs: str, channel: str, artifacts: Optional[List[Dict[str, Any]]], data_dir: str
) -> Optional[str]:
    """The AIA FITS that goes with a model file: the ``pair`` of its artifact event, else the newest
    ``aia_lev15_<ch>_*.fits`` in ``<data_dir>/aia`` or beside the model. Only files under the data folder count."""
    real = os.path.realpath(model_abs)
    for a in reversed(artifacts or []):
        pair = a.get("pair")
        if pair and a.get("path") and os.path.realpath(a["path"]) == real:
            full = pair if os.path.isabs(pair) else os.path.join(data_dir, pair)
            if os.path.isfile(full) and _under(full, data_dir):
                return full
    for folder in (os.path.join(data_dir, "aia"), os.path.dirname(model_abs)):
        hits = [h for h in glob.glob(os.path.join(folder, f"aia_lev15_{channel}_*.fits")) if _under(h, data_dir)]
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def _shrink(img: np.ndarray, size: int):
    """Block mean to at most ``size`` pixels a side (NaN-aware). Returns (array, factor); the origin pixel stays put."""
    n = max(img.shape)
    f = max(1, -(-n // size))
    if f == 1:
        return img.astype(np.float32), 1
    ny, nx = (-(-s // f) * f for s in img.shape)
    padded = np.full((ny, nx), np.nan)
    padded[: img.shape[0], : img.shape[1]] = img
    blocks = padded.reshape(ny // f, f, nx // f, f)
    with np.errstate(all="ignore"):
        out = np.nanmean(blocks, axis=(1, 3))
    return out.astype(np.float32), f


def _pack(img: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(img, dtype="<f4").tobytes()).decode("ascii")


def _num(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _stretch_range(*imgs) -> Optional[List[float]]:
    """The same 5 and 99.7 percentiles of the positive pixels of both images that aia.compare's own PNG uses."""
    vals = np.concatenate([i[np.isfinite(i) & (i > 0)] for i in imgs])
    if not vals.size:
        return None
    lo, hi = np.percentile(vals, [5, 99.7])
    return [float(lo), float(hi)]


def build(model_path: str, aia_path: Optional[str], channel: str, size: int) -> Dict[str, Any]:
    """The compare payload for one model file (and its AIA file, when there is one). Cached by file stamps."""
    stamp = lambda p: (os.path.realpath(p), os.stat(p).st_mtime_ns, os.stat(p).st_size) if p else None  # noqa: E731
    key = (stamp(model_path), stamp(aia_path), channel, size)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    from astropy.io import fits

    with fits.open(model_path) as h:
        model = np.asarray(h[0].data, float)
        hdr = h[0].header.copy()
    metrics: Optional[Dict[str, Any]] = None
    aia = None
    prep = None
    if aia_path:
        from fluxpype.science import aia as aia_mod

        with tempfile.TemporaryDirectory() as tmp:
            out, summary = aia_mod.compare(model_path, aia_path, int(channel), os.path.join(tmp, "c.fits"), png=False)
            with fits.open(out) as c:
                aia = np.asarray(c["AIA_REPROJ"].data, float)
        with fits.open(aia_path) as a:
            prep = a[0].header.get("PREPSTEP")
        metrics = {k: _num(v) for k, v in summary.items()}
        metrics["pixels_compared"] = int(summary["pixels_compared"])
    small, f = _shrink(model, size)
    wcs = {k: hdr[k] for k in WCS_KEYS if k in hdr}
    for k in ("CRPIX1", "CRPIX2"):
        if k in wcs:
            wcs[k] = (float(wcs[k]) - 0.5) / f + 0.5
    for k in ("CDELT1", "CDELT2"):
        if k in wcs:
            wcs[k] = float(wcs[k]) * f
    ashrunk = _shrink(aia, size)[0] if aia is not None else None
    payload = {
        "channel": channel,
        "shape": [int(small.shape[0]), int(small.shape[1])],
        "binning": f,
        "full_shape": [int(model.shape[0]), int(model.shape[1])],
        "wcs": wcs,
        "unit": hdr.get("BUNIT"),
        "model": {"file": os.path.basename(model_path), "data": _pack(small)},
        "aia": (
            {"file": os.path.basename(aia_path), "data": _pack(ashrunk), "prep": prep} if ashrunk is not None else None
        ),
        "metrics": metrics,
        "stretch": _stretch_range(small, ashrunk) if ashrunk is not None else _stretch_range(small),
    }
    with _lock:
        _cache[key] = payload
        while len(_cache) > 6:
            _cache.popitem(last=False)
    return payload


def for_task(
    products: List[Dict[str, Any]],
    artifacts: Optional[List[Dict[str, Any]]],
    data_dir: str,
    channel: Optional[str],
    size,
    cr=None,
    arrays: bool = True,
) -> Dict[str, Any]:
    """The body of GET /api/tasks/{id}/compare (without the task id). Never invents an AIA image."""
    models = find_model_files(products)
    out: Dict[str, Any] = {"channels": sorted(models, key=int), "channel": None, "available": False, "note": None}
    if not models:
        out["note"] = (
            "This task has no FLUXlight EUV image (cr..._aia<channel>.fits). Run FLUXlight with the EUV renderer "
            "(fluxlight_euv) first."
        )
        return out
    ch = str(channel) if channel else ("193" if "193" in models else out["channels"][0])
    out["channel"] = ch
    if ch not in models:
        out["note"] = f"No model image for channel {ch}; this task has {', '.join(out['channels'])}."
        return out
    model_abs = os.path.join(data_dir, models[ch])
    aia_abs = find_aia_file(model_abs, ch, artifacts, data_dir)
    if not aia_abs:
        out["aia_missing"] = True
        out["note"] = (
            "The AIA image is not on this machine, so only the FLUXlight image is shown. AIA is downloaded where the "
            "network allows it (not in the cloud container): "
            + FETCH_HINT.format(cr=cr if cr is not None else "<CR>", ch=ch)
        )
    body = build(model_abs, aia_abs, ch, clamp_size(size))
    out.update(body, available=True)
    if not arrays:
        for k in ("model", "aia"):
            if out.get(k):
                out[k] = {kk: v for kk, v in out[k].items() if kk != "data"}
    return out


# ---------------------------------------------------------------------------------------------- picture for agents


def unpack(entry: Dict[str, Any], shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(entry["data"]), dtype="<f4").reshape(shape)


def side_by_side_png(body: Dict[str, Any]) -> bytes:
    """FLUXlight | AIA | log10 model/AIA as one PNG, on the shared log stretch of the payload (Pillow, numpy)."""
    import io

    from PIL import Image

    shape = body["shape"]
    lo, hi = body["stretch"]

    def gray(img):
        with np.errstate(all="ignore"):
            v = (np.log10(np.clip(np.nan_to_num(img), lo, None)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return (np.clip(v, 0, 1) * 255).astype(np.uint8)[::-1]  # north up

    panes = [np.stack([gray(unpack(body["model"], shape))] * 3, -1)]
    if body.get("aia"):
        m, a = unpack(body["model"], shape), unpack(body["aia"], shape)
        panes.append(np.stack([gray(a)] * 3, -1))
        with np.errstate(all="ignore"):
            lr = np.where((m > 0) & (a > 0), np.log10(m / a), np.nan)
        t = np.clip(np.nan_to_num(lr) / 2.0, -1, 1)  # -2..2 decades: blue (model low) through white to red (model high)
        rgb = np.stack([255 - 255 * np.clip(-t, 0, 1), 255 * (1 - np.abs(t)), 255 - 255 * np.clip(t, 0, 1)], -1)
        rgb[~np.isfinite(lr)] = 0
        panes.append(rgb.astype(np.uint8)[::-1])
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(panes, axis=1)).save(buf, "PNG")
    return buf.getvalue()
