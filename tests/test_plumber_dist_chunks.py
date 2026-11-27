"""The web bundle stays split: page routes are lazy chunks and the main chunk stays small (A4-10)."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "fluxpype" / "plumber" / "web"
ASSETS = WEB / "dist" / "assets"
MAIN_LIMIT = 450_000


def _main_chunk() -> Path:
    html = (WEB / "dist" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'src="/assets/(index-[^"]+\.js)"', html)
    assert m, "dist/index.html has no main script"
    return ASSETS / m.group(1)


def test_main_chunk_under_limit():
    size = _main_chunk().stat().st_size
    assert size < MAIN_LIMIT, f"main chunk is {size} bytes; keep pages lazy or rebuild web/dist"


def test_lazy_chunks_exist():
    main = _main_chunk().read_text(encoding="utf-8")
    names = set(re.findall(r'"\./([A-Za-z0-9_-]+-[A-Za-z0-9_-]{8}\.js)"', main))
    assert names, "main chunk imports no lazy chunks"
    missing = [n for n in names if not (ASSETS / n).is_file()]
    assert not missing, f"dist is missing lazy chunks: {missing}"


def test_chunk_warning_limit_not_raised():
    assert "chunkSizeWarningLimit" not in (WEB / "vite.config.ts").read_text(encoding="utf-8")
