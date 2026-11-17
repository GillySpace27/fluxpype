"""relax_movie.py stitches simple_relaxer's frames, in step order, into a GIF beside them."""

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from fluxpype.plotting import relax_movie  # noqa: E402


def test_gif_in_step_order(tmp_path):
    for step, shade in [(100, 200), (5, 0), (25, 100)]:
        Image.new("RGB", (40, 30), (shade, shade, shade)).save(tmp_path / f"cr2150_f100_hmi_relax_s{step:05d}.png")
    (tmp_path / "unrelated.png").write_bytes(b"x")
    frames = relax_movie.find_frames(str(tmp_path))
    assert [p.rsplit("_s", 1)[1] for p in frames] == ["00005.png", "00025.png", "00100.png"]
    made = relax_movie.make_movie(str(tmp_path / "cr2150_f100_hmi_relax_s00025.png"), mp4=False)
    gif = tmp_path / "cr2150_f100_hmi_relax_movie.gif"
    assert made == [str(gif)]
    with Image.open(gif) as im:
        assert im.n_frames == 3


def test_no_frames(tmp_path):
    with pytest.raises(SystemExit):
        relax_movie.make_movie(str(tmp_path))
