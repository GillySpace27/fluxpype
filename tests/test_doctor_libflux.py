"""flux-doctor reports libflux's optimization level from its DWARF producer, and only warns when it can tell."""

from fluxpype import doctor

O0 = b"\0GNU C17 12.2.0 -mtune=generic -march=x86-64 -g -fPIC\0"
O2 = b"\0GNU C17 12.2.0 -mtune=generic -march=x86-64 -g -fPIC -O2\0"


def test_opt_level_o0_when_producer_has_no_o_flag():
    assert doctor.libflux_opt_level(b"junk" + O0 + O0) == "O0"


def test_opt_level_optimized_for_o1_o2_o3_os():
    for flag in (b"-O1", b"-O2", b"-O3", b"-Os", b"-Ofast"):
        assert doctor.libflux_opt_level(O0.replace(b"-fPIC", b"-fPIC " + flag)) == "optimized"


def test_opt_level_unknown_without_gcc_producer():
    assert doctor.libflux_opt_level(b"no debug info here") is None
    assert doctor.libflux_opt_level(b"\0Apple clang version 15.0.0\0") is None


def test_check_warns_on_o0_with_exact_make_line_and_trap(tmp_path):
    f = tmp_path / "libflux.a"
    f.write_bytes(O0)
    ok, detail, fix = doctor.libflux_check([str(f)])
    assert ok is None and "O0" in detail
    assert 'make everything CFLAGS="-I../include -g -fPIC -O2"' in fix
    assert "/usr/local/lib/libflux.a" in fix and "-L/usr/local/lib" in fix


def test_check_passes_when_optimized(tmp_path):
    f = tmp_path / "libflux.a"
    f.write_bytes(O2)
    assert doctor.libflux_check([str(f)])[0] is True


def test_check_unreadable_gives_hint_not_warning_text(tmp_path):
    f = tmp_path / "libflux.a"
    f.write_bytes(b"no debug info")
    ok, detail, fix = doctor.libflux_check([str(f)])
    assert ok == "info" and "unknown" in detail and fix.startswith("hint:") and "-O2" in fix


def test_check_missing_library_is_a_hint():
    ok, _, fix = doctor.libflux_check([])
    assert ok == "info" and fix.startswith("hint:")


def test_mixed_archive_is_not_optimized(tmp_path):
    assert doctor.libflux_opt_level(O2 + O0) == "mixed"
    f = tmp_path / "libflux.a"
    f.write_bytes(O2 + O0)
    ok, _, fix = doctor.libflux_check([str(f)])
    assert ok is None and "without optimization" in fix


def test_bare_O_is_optimized_and_explicit_O0_is_not():
    assert doctor.libflux_opt_level(O0.replace(b"-fPIC", b"-fPIC -O")) == "optimized"
    assert doctor.libflux_opt_level(O0.replace(b"-fPIC", b"-fPIC -O2 -O0")) == "O0"


def test_producer_without_switches_is_unknown():
    assert doctor.libflux_opt_level(b"\0GNU C17 12.2.0\0") is None
    # readable producers still count when another one records no switches
    assert doctor.libflux_opt_level(b"\0GNU C17 12.2.0\0" + O2) == "optimized"


def test_candidates_include_config_fl_prefix_after_usr_local(tmp_path, monkeypatch):
    lib = tmp_path / "pfx" / "lib"
    lib.mkdir(parents=True)
    (lib / "libflux.a").write_bytes(O2)
    monkeypatch.delenv("FL_PREFIX", raising=False)
    monkeypatch.setattr(doctor, "_config_fl_prefix", lambda: str(tmp_path / "pfx") + "  # comment")
    cands = doctor._libflux_candidates()
    assert str(lib / "libflux.a") in cands
    monkeypatch.setenv("FL_PREFIX", str(tmp_path / "pfx"))
    assert doctor._libflux_candidates().count(str(lib / "libflux.a")) == 1


def test_verdict_comes_from_the_first_linked_archive(tmp_path):
    first, other = tmp_path / "a.a", tmp_path / "b.a"
    first.write_bytes(O2)
    other.write_bytes(O0)
    ok, detail, _ = doctor.libflux_check([str(first), str(other)])
    assert ok is True and "b.a: O0" in detail
    ok, detail, fix = doctor.libflux_check([str(other), str(first)])
    assert ok is None and "a.a: optimized" in detail and "Link-order trap" in fix


def test_advice_uses_known_checkout_else_generic(tmp_path, monkeypatch):
    assert "in your fluxon-mhd checkout" in doctor._advice("O0", None)
    co = tmp_path / "fluxon-mhd"
    (co / "lib").mkdir(parents=True)
    (co / "Makefile").write_text("")
    monkeypatch.setenv("FL_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setattr(doctor, "_config_fl_prefix", lambda: "")
    assert doctor._checkout_dir() == str(co)
    assert f"cd {co}" in doctor._advice("O0", str(co))


def test_main_prints_info_for_hints_without_changing_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "checks", lambda: iter([("x", True, "d", ""), ("libflux", "info", "d2", "hint: h")]))
    try:
        doctor.main()
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "[info] libflux: d2" in out and "hint: h" in out and "[warn]" not in out and "fix:" not in out
