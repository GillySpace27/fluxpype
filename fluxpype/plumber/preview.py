"""
Previews of files in the data folder, for the Cache page's viewer.

``describe`` decides how a file can be shown and returns what the UI needs: the first lines of a text
file, the headers of a FITS file (read here in pure Python, no astropy), or just its size. Pickles are
never opened, because unpickling runs code. ``render_fits`` draws a FITS image to a PNG with the job
Python (which has numpy, astropy and matplotlib), and keeps the result in ``<home>/previews``.
"""

import gzip
import hashlib
import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}
PDF = {".pdf": "application/pdf"}
FITS_EXT = (".fits", ".fts", ".fit", ".fits.gz", ".fts.gz")
TEXT_LIMIT = 64 * 1024
TEXT_LINES = 400
MAX_HDUS = 4
MAX_CARDS = 2000
RENDER_TIMEOUT = 90.0
_RENDERS = threading.Semaphore(2)  # a grid of FITS thumbnails asks for many renders at once; each one is a matplotlib process
KEEP_RENDERS = 300


def kind_of(path: str) -> str:
    low = path.lower()
    ext = os.path.splitext(low)[1]
    if ext in IMAGE_TYPES:
        return "image"
    if ext in PDF:
        return "pdf"
    if low.endswith(FITS_EXT):
        return "fits"
    if ext in (".pkl", ".pickle"):
        return "pickle"
    return "text_or_binary"


def media_type(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    return IMAGE_TYPES.get(ext) or PDF.get(ext) or "application/octet-stream"


def _looks_binary(chunk: bytes) -> bool:
    if not chunk:
        return False
    if b"\x00" in chunk[:8192]:
        return True
    sample = chunk[:8192]
    odd = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return odd / len(sample) > 0.1


def read_text_head(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        chunk = f.read(TEXT_LIMIT + 1)
    if _looks_binary(chunk):
        return {"binary": True}
    more = len(chunk) > TEXT_LIMIT
    text = chunk[:TEXT_LIMIT].decode("utf-8", errors="replace")
    lines = text.splitlines()
    if more and lines:
        lines = lines[:-1]  # the last line is probably cut
    cut = len(lines) > TEXT_LINES
    return {"binary": False, "lines": lines[:TEXT_LINES], "truncated": more or cut}


# ------------------------------------------------------------------ FITS headers (pure Python)
def _card(raw: bytes) -> Optional[Dict[str, str]]:
    s = raw.decode("ascii", errors="replace")
    key = s[:8].strip()
    if not key and not s.strip():
        return None
    if s[8:10] == "= ":
        body = s[10:]
        if body.lstrip().startswith("'"):  # string value, '' escapes a quote
            start = body.index("'") + 1
            i, out = start, []
            while i < len(body):
                if body[i] == "'":
                    if i + 1 < len(body) and body[i + 1] == "'":
                        out.append("'")
                        i += 2
                        continue
                    break
                out.append(body[i])
                i += 1
            value = "".join(out).rstrip()
            rest = body[i + 1 :]
            comment = rest.split("/", 1)[1].strip() if "/" in rest else ""
        else:
            value, _, comment = body.partition("/")
            value, comment = value.strip(), comment.strip()
        return {"key": key, "value": value, "comment": comment}
    return {"key": key, "value": "", "comment": s[8:].rstrip()}


def _data_bytes(cards: List[Dict[str, str]]) -> int:
    h = {c["key"]: c["value"] for c in cards}
    try:
        naxis = int(h.get("NAXIS", "0"))
        if naxis == 0:
            return 0
        n = 1
        for i in range(1, naxis + 1):
            n *= int(h.get(f"NAXIS{i}", "0"))
        bits = abs(int(h.get("BITPIX", "8")))
        pcount, gcount = int(h.get("PCOUNT", "0")), int(h.get("GCOUNT", "1"))
        size = bits // 8 * gcount * (pcount + n)
    except ValueError:
        return -1
    return (size + 2879) // 2880 * 2880


def fits_headers(path: str) -> List[Dict[str, Any]]:
    """Headers of the first few HDUs. Raises ValueError if this is not a FITS file."""
    opener = gzip.open if path.lower().endswith(".gz") else open
    hdus: List[Dict[str, Any]] = []
    with opener(path, "rb") as f:
        while len(hdus) < MAX_HDUS:
            cards: List[Dict[str, str]] = []
            ended = False
            while not ended:
                block = f.read(2880)
                if len(block) < 2880:
                    if not hdus and not cards:
                        raise ValueError("empty file")
                    return hdus if not cards else hdus + [{"cards": cards, "incomplete": True}]
                if not hdus and not cards and not block.startswith((b"SIMPLE  =", b"XTENSION=")):
                    raise ValueError("not a FITS file")
                for i in range(0, 2880, 80):
                    raw = block[i : i + 80]
                    if raw[:8] == b"END     ":
                        ended = True
                        break
                    c = _card(raw)
                    if c and len(cards) < MAX_CARDS:
                        cards.append(c)
            h = {c["key"]: c["value"] for c in cards}
            dims = []
            try:
                dims = [int(h.get(f"NAXIS{i}", "0")) for i in range(1, int(h.get("NAXIS", "0")) + 1)]
            except ValueError:
                pass
            hdus.append(
                {
                    "name": h.get("EXTNAME") or ("PRIMARY" if not hdus else h.get("XTENSION", "EXT")),
                    "dims": dims,
                    "bitpix": h.get("BITPIX"),
                    "cards": cards,
                }
            )
            skip = _data_bytes(cards)
            if skip < 0:
                break
            if skip:
                if hasattr(f, "seek") and not path.lower().endswith(".gz"):
                    f.seek(skip, os.SEEK_CUR)
                else:
                    while skip > 0:
                        got = f.read(min(skip, 1 << 20))
                        if not got:
                            return hdus
                        skip -= len(got)
    return hdus


# ------------------------------------------------------------------ describe
def describe(path: str, rel: str) -> Dict[str, Any]:
    st = os.stat(path)
    out: Dict[str, Any] = {"path": rel, "name": os.path.basename(rel), "size": st.st_size, "mtime": st.st_mtime}
    kind = kind_of(path)
    if kind in ("image", "pdf"):
        return {**out, "view": kind, "media_type": media_type(path)}
    if kind == "pickle":
        return {
            **out,
            "view": "none",
            "note": "Python pickle (a saved PFSS solution). Plumber does not open pickles: loading one runs code.",
        }
    if kind == "fits":
        try:
            hdus = fits_headers(path)
        except (OSError, ValueError, EOFError) as e:
            return {**out, "view": "none", "note": f"could not read the FITS header: {e}"}
        renderable = any(len(h.get("dims", [])) >= 2 and all(d > 0 for d in h["dims"][:2]) for h in hdus)
        return {**out, "view": "fits", "hdus": hdus, "renderable": renderable}
    head = read_text_head(path)
    if head["binary"]:
        return {**out, "view": "none", "note": "binary file; no preview"}
    return {**out, "view": "text", "lines": head["lines"], "truncated": head["truncated"]}


# ------------------------------------------------------------------ FITS rendering (job Python)
RENDER_SCRIPT = r"""
import json, sys
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
with fits.open(src, memmap=False) as hd:
    hdu = next((h for h in hd if h.data is not None and getattr(h.data, "ndim", 0) >= 2), None)
    if hdu is None:
        sys.exit("no image data in this file")
    d = np.asarray(hdu.data, dtype=float)
    name = hdu.name
while d.ndim > 2:
    d = d[0]
fin = d[np.isfinite(d)]
signed = bool(fin.size and fin.min() < 0)
if fin.size == 0:
    lo, hi = 0.0, 1.0
elif signed:
    hi = float(np.percentile(np.abs(fin), 99.5)) or 1.0
    lo = -hi
else:
    lo, hi = (float(x) for x in np.percentile(fin, [0.5, 99.5]))
ny, nx = d.shape
scale = 1400.0 / max(nx, ny)
fig = plt.figure(figsize=(max(nx * scale, 50) / 100, max(ny * scale, 50) / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.imshow(d, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
fig.savefig(dst)
print(json.dumps({
    "hdu": name, "shape": [int(ny), int(nx)], "vmin": lo, "vmax": hi,
    "min": float(fin.min()) if fin.size else None, "max": float(fin.max()) if fin.size else None,
    "nan": int(d.size - fin.size),
}))
"""


class RenderError(RuntimeError):
    pass


def render_fits(path: str, home: str, python: str) -> Dict[str, Any]:
    """PNG of the first 2-D image in `path`, cached by path, size and mtime. Returns {png, stats}."""
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
    folder = os.path.join(str(home), "previews")
    os.makedirs(folder, exist_ok=True)
    png, meta = os.path.join(folder, key + ".png"), os.path.join(folder, key + ".json")
    if os.path.isfile(png) and os.path.isfile(meta):
        with open(meta, encoding="utf-8") as f:
            return {"png": png, "stats": json.load(f)}
    tmp = png + f".tmp{os.getpid()}.png"
    try:
        with _RENDERS:
            if os.path.isfile(png) and os.path.isfile(meta):  # another request rendered it while this one waited
                with open(meta, encoding="utf-8") as f:
                    return {"png": png, "stats": json.load(f)}
            r = subprocess.run(
                [python, "-c", RENDER_SCRIPT, path, tmp],
                capture_output=True,
                text=True,
                timeout=RENDER_TIMEOUT,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
    except subprocess.TimeoutExpired:
        raise RenderError(f"rendering took longer than {RENDER_TIMEOUT:.0f} s")
    except OSError as e:
        raise RenderError(f"cannot start {python}: {e}")
    if r.returncode != 0 or not os.path.isfile(tmp):
        lines = [ln for ln in (r.stderr or "").splitlines() if ln.strip()]
        raise RenderError((lines[-1] if lines else "render failed")[:300] + f" (with {python})")
    try:
        stats = json.loads((r.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        stats = {}
    os.replace(tmp, png)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    _prune(folder)
    return {"png": png, "stats": stats}


def _prune(folder: str) -> None:
    try:
        pngs = sorted(
            (e for e in os.scandir(folder) if e.name.endswith(".png") and ".tmp" not in e.name),
            key=lambda e: e.stat().st_mtime,
        )
    except OSError:
        return
    for e in pngs[:-KEEP_RENDERS]:
        for p in (e.path, e.path[:-4] + ".json"):
            try:
                os.remove(p)
            except OSError:
                pass
