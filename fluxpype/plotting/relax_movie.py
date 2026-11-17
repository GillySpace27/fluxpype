"""Stitch the relaxation movie frames that simple_relaxer writes into an animated GIF (and an MP4 when ffmpeg is
on the PATH).

The frames are ``<world>_relax_s<step>.png`` in ``imgs/relax/<world>/`` (relax_pfss_world.pm). The movie goes
beside them as ``<world>_relax_movie.gif`` / ``.mp4``.

    python relax_movie.py <frame folder, or any one frame in it> [--fps 6]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List

_FRAME = re.compile(r"^(?P<tag>.+)_relax_s(?P<step>\d+)\.png$")


def find_frames(folder: str) -> List[str]:
    """The frames in ``folder``, in step order."""
    hits = []
    for name in os.listdir(folder):
        m = _FRAME.match(name)
        if m:
            hits.append((int(m.group("step")), os.path.join(folder, name)))
    return [p for _, p in sorted(hits)]


def write_gif(frames: List[str], out: str, fps: float, hold_last: int = 3, max_px: int = 720) -> str:
    """An animated GIF of ``frames``; the last frame is held ``hold_last`` times as long so the loop reads."""
    from PIL import Image

    imgs = []
    for p in frames:
        im = Image.open(p).convert("RGB")
        if max(im.size) > max_px:
            s = max_px / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        imgs.append(im.quantize(colors=255, method=Image.Quantize.MEDIANCUT))
    ms = int(round(1000 / fps))
    durations = [ms] * len(imgs)
    durations[-1] = ms * hold_last
    tmp = out + ".part"
    imgs[0].save(tmp, format="GIF", save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
    os.replace(tmp, out)
    return out


def write_mp4(frames: List[str], out: str, fps: float) -> str:
    """An H.264 MP4 of ``frames`` via ffmpeg (frames are linked into a numbered sequence first)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not on PATH")
    with tempfile.TemporaryDirectory() as td:
        for i, p in enumerate(frames):
            dst = os.path.join(td, f"f{i:05d}.png")
            try:
                os.symlink(os.path.abspath(p), dst)
            except OSError:
                shutil.copy(p, dst)
        tmp = out + ".part.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(td, "f%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            tmp,
        ]
        subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    return out


def make_movie(target: str, fps: float = 6.0, mp4: bool = True) -> List[str]:
    folder = target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target))
    frames = find_frames(folder)
    if not frames:
        raise SystemExit(f"relax_movie: no *_relax_s<step>.png frames in {folder}")
    tag = _FRAME.match(os.path.basename(frames[0])).group("tag")
    base = os.path.join(folder, f"{tag}_relax_movie")
    made = [write_gif(frames, base + ".gif", fps)]
    if mp4 and shutil.which("ffmpeg"):
        try:
            made.append(write_mp4(frames, base + ".mp4", fps))
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"relax_movie: MP4 skipped ({e}); the GIF is fine", file=sys.stderr)
    for p in made:
        print(f"\twrote {p} ({len(frames)} frames)")
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="the frame folder, or any frame in it")
    ap.add_argument("--fps", type=float, default=6.0, help="frames per second (default 6)")
    ap.add_argument("--no-mp4", action="store_true", help="write only the GIF")
    a = ap.parse_args(argv)
    make_movie(a.target, a.fps, mp4=not a.no_mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
