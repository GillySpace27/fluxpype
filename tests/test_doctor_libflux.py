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
    assert ok is None and "unknown" in detail and fix.startswith("hint:") and "-O2" in fix


def test_check_missing_library_is_a_hint():
    ok, _, fix = doctor.libflux_check([])
    assert ok is None and fix.startswith("hint:")
