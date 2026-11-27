"""config_io: parsing the real config.ini, task expansion, comment-preserving overrides."""

import pytest

from fluxpype.plumber import config_io as c
from fluxpype.plumber import paths

CHECKOUT = str(paths.checkout_root())


@pytest.fixture(scope="module")
def shipped():
    return c.read_file(paths.shipped_config())


def test_parses_every_shipped_profile(shipped):
    assert "quickstart" in shipped.profiles and len(shipped.profiles) > 10
    eff = shipped.effective("quickstart")
    assert eff["batch_name"].value == "quickstart" and eff["batch_name"].section == "quickstart"
    assert eff["batch_name"].overrides_default is True
    assert eff["base_dir"].section == "DEFAULT"
    assert eff["relax_threshold"].value == "1.5"
    # inline comments become help text, and are not part of the value
    assert "Percentage threshold" in eff["relax_threshold"].comment
    assert "#" not in eff["rotations"].value


def test_commented_out_lines_are_not_settings(shipped):
    assert "wind_only" not in shipped.effective("velocityplots")  # `; wind_only = 0` is a comment
    assert "flux_perl_lib" not in shipped.defaults  # the shipped line starts with ';'


def test_task_product_in_runner_order():
    eff = c.parse(
        "[DEFAULT]\nrotations=[1, 2]\nfluxon_count=[10, 20]\nadapts=[0, 1]\nflow_method=[Parker, wsa]\nrun_script=x\n[p]\n"  # noqa: E501
    ).effective("p")
    plan = c.expand_tasks(eff)
    assert len(plan.tasks) == 16 and plan.errors == []
    # adapts outermost, then rotations, then fluxon_count, then methods (config_runner.run loop order)
    assert plan.tasks[:4] == [
        ("0", "1", "10", "parker"),
        ("0", "1", "10", "wsa"),
        ("0", "1", "20", "parker"),
        ("0", "1", "20", "wsa"),
    ]
    assert plan.tasks[8][0] == "1"


def test_ranges_and_single_values():
    eff = c.parse(
        "[DEFAULT]\nrotations=(2100, 2110, 5)\nfluxon_count=7\nadapts=[0]\nflow_method=parker\nrun_script=x\n[p]\n"
    ).effective("p")
    plan = c.expand_tasks(eff)
    assert plan.axes["rotations"] == ["2100", "2105"]
    assert plan.axes["fluxon_count"] == ["7"] and plan.axes["flow_method"] == ["parker"]


def test_matches_pipe_helper_semantics(fake_science):
    """The local helpers must agree with pipe_helper's (imported with a fake science stack)."""
    from fluxpype import pipe_helper as ph

    for value in ("[1, 2, 3]", "(2100, 2110, 5)", "5", "[a,]", "[]"):
        assert c.parse_list_or_range(value) == ph.parse_list_or_range(value)
    for value in ("[Parker, WSA]", "parker", "[x,]", ""):
        assert c.parse_list(value) == ph.parse_list(value)


def test_trailing_comma_is_flagged_and_dropped():
    eff = c.parse(
        "[DEFAULT]\nrotations=[1]\nfluxon_count=[10]\nadapts=[0]\nflow_method=[parker,]\nrun_script=x\n[p]\n"
    ).effective("p")
    plan = c.expand_tasks(eff)
    assert plan.axes["flow_method"] == ["parker"]
    assert any("flow_method" in w and "empty item" in w for w in plan.warnings)
    assert len(plan.tasks) == 1


def test_profile_without_flow_method_cannot_run(shipped):
    plan = c.expand_tasks(shipped.effective("Real_Run"))
    assert plan.tasks == [] and any("flow_method" in e for e in plan.errors)


def test_overrides_keep_comments_and_touch_only_their_lines(shipped):
    text = paths.shipped_config().read_text()
    out = c.apply_overrides(text, "quickstart", {"relax_threshold": "2.5", "concurrency": "2", "max_cycles": "9"})
    before, after = text.splitlines(), out.splitlines()
    changed = [line for line in after if line not in before]
    # an existing profile line is rewritten in place; keys only in DEFAULT are appended to the profile
    assert sorted(changed) == ["concurrency = 2", "max_cycles = 9", "relax_threshold = 2.5"]
    assert len(after) == len(before) + 2
    # DEFAULT's commented relax_threshold line is untouched
    assert "relax_threshold = 1.5  # Percentage threshold for stopping relaxation" in out
    eff = c.parse(out).effective("quickstart")
    assert eff["relax_threshold"].value == "2.5" and eff["max_cycles"].value == "9"
    assert c.parse(out).defaults["relax_threshold"].value == "1.5"


def test_override_of_an_existing_profile_line_keeps_its_inline_comment():
    text = "[DEFAULT]\nx = 1\nrun_script = s\n[p]\nrotations  = [2150] # a comment\nx = 5  # keep me\n"
    out = c.apply_overrides(text, "p", {"x": "7", "rotations": "[2200]"})
    assert "x = 7  # keep me" in out
    assert "rotations  = [2200] # a comment" in out


@pytest.mark.parametrize(
    "key,value",
    [
        ("run_script", "evil.pdl"),
        ("base_dir", "/tmp"),
        ("data_dir", "/tmp"),
        ("python_dir", "x"),
        ("rc_path", "x"),
        ("fl_prefix", "x"),
        ("pl_prefix", "x"),
        ("fl_mhdlib", "x"),
        ("working_dir", "x"),
        ("flux_perl_lib", "x"),
        ("not_a_setting", "1"),
        ("max_cycles", "1 # sneaky"),
        ("max_cycles", "${base_dir}"),
        ("max_cycles", "1;2"),
        ("max_cycles", "1\n[x]"),
        ("max_cycles", "__import__"),
        ("max_cycles", "abs(1)"),
        ("max_cycles", "a[0]"),
        ("max_cycles", "'x'"),
        ("max_cycles", "../x"),
        ("batch_name", "a/b"),
        ("rotations", "a" * 300),
    ],
)
def test_rejected_overrides(key, value):
    text = paths.shipped_config().read_text()
    with pytest.raises(c.ConfigError):
        c.apply_overrides(text, "quickstart", {key: value})


@pytest.mark.parametrize(
    "value", ["[2200, 2201]", "(2100, 2110, 5)", "0.25", "parker", "[0.3, 0.1]", "1e3", "-1", "my batch.v2"]
)
def test_accepted_override_values(value):
    text = "[DEFAULT]\nk = 1\nbatch_name = b\nrun_script = s\n[p]\n"
    key = "batch_name" if value == "my batch.v2" else "k"
    assert c.parse(c.apply_overrides(text, "p", {key: value})).effective("p")[key].value == value


def test_unknown_profile():
    with pytest.raises(c.ConfigError):
        c.apply_overrides("[DEFAULT]\na=1\n", "nope", {"a": "2"})


def test_snapshot_pins_base_dir_and_is_otherwise_verbatim():
    text = paths.shipped_config().read_text()
    snap = c.make_snapshot(text, "quickstart", {}, CHECKOUT)
    diff = [(a, b) for a, b in zip(text.splitlines(), snap.splitlines()) if a != b]
    assert len(diff) == 1 and diff[0][0].startswith("base_dir = auto")
    assert diff[0][1].startswith("base_dir = " + c.auto_base_dir(CHECKOUT))
    assert len(text.splitlines()) == len(snap.splitlines())


def test_resolved_data_dir_matches_auto_base_dir(shipped):
    eff = shipped.effective("quickstart")
    assert c.resolved_path(eff, "data_dir", CHECKOUT).startswith(c.auto_base_dir(CHECKOUT))


def test_instrument_names():
    assert c.task_instrument("0") == "hmi" and c.task_instrument("1") == "adapt"


# ---------------------------------------------------------------- "Compare with AIA" keys (audit A4-09)

AIA_KEYS = ("observation_date", "mag_file", "aia_channels")


def test_aia_keys_default_to_unset_in_every_profile(shipped):
    for name in shipped.profiles:
        eff = shipped.effective(name)
        assert all(k in eff for k in AIA_KEYS), name
        assert eff["observation_date"].value == "None" and eff["mag_file"].value == "None"


def test_aia_keys_defaults_leave_science_config_as_before():
    from fluxpype.config import convert_value

    assert (
        convert_value("None") is None
    )  # observation_date / mag_file read as None: observer_lon and header date unchanged
    assert convert_value("[193]") == [193]


def test_aia_overrides_validate_and_survive_all_three_parsers(shipped, tmp_path, monkeypatch):
    import configparser

    from fluxpype import config as flux_config

    eff = shipped.effective("light")
    keys = set(eff)
    ov = {
        "observation_date": "2014-05-18T20:00:00",
        "aia_channels": "[171, 193]",
        "mag_file": "batches/b/mags/CR2150_r4_hmi.fits",
    }
    ok = {k: c.validate_override(keys, k, v) for k, v in ov.items()}
    assert ok == ov
    for bad in ("2014-05-18;rm", "$(x)", "a" * 201):
        with pytest.raises(c.ConfigError):
            c.validate_override(keys, "observation_date", bad)
    snap = c.make_snapshot(shipped.text, "light", ok, str(tmp_path))
    # parser 1: config_io
    got = c.parse(snap).effective("light")
    assert got["observation_date"].value == "2014-05-18T20:00:00" and got["aia_channels"].value == "[171, 193]"
    # parser 2: config.configurations (configparser + convert_value)
    p = tmp_path / "snap.ini"
    p.write_text(snap)
    monkeypatch.setenv("FLUXPYPE_CONFIG_FILE", str(p))
    cfg = flux_config.configurations("light")
    assert cfg["observation_date"] == "2014-05-18T20:00:00" and cfg["aia_channels"] == [171, 193]
    assert cfg["mag_file"] == ov["mag_file"]
    # parser 3: the Perl side strips "#" comments and reads key = value with Config::IniFiles; check the same
    # cleaned text still carries the values (Config::IniFiles itself is not installed in CI).
    cleaned = "\n".join(line.split("#", 1)[0].rstrip() for line in snap.splitlines())
    raw = configparser.ConfigParser(interpolation=None)
    raw.read_string(cleaned)
    assert raw["light"]["observation_date"] == "2014-05-18T20:00:00"
