"""config_io: parsing the real config.ini, task expansion, comment-preserving overrides."""

import pytest

from fluxpype.studio import config_io as c
from fluxpype.studio import paths

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
        "[DEFAULT]\nrotations=[1, 2]\nfluxon_count=[10, 20]\nadapts=[0, 1]\nflow_method=[Parker, wsa]\nrun_script=x\n[p]\n"
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
    changed = [l for l in after if l not in before]
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
