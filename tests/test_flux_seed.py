"""flux_seed (audit A1-05): parsing, the Julian-date seed, and the env the FLUX Perl processes get."""

import pytest

from fluxpype import flux_seed as fs

# 2000-01-01 12:00:00 UTC is JD 2451545.0 exactly; its Unix time is 946728000.
J2000_UNIX = 946728000.0


def test_julian_date_of_j2000():
    assert fs.julian_date(J2000_UNIX) == pytest.approx(2451545.0)


@pytest.mark.parametrize("raw", [None, "", " ", "0", "fixed", "FIXED", "off"])
def test_fixed_spellings_give_zero(raw):
    assert fs.parse_seed(raw) == 0


def test_integer_seed_is_taken_as_is():
    assert fs.parse_seed("12345") == 12345
    assert fs.parse_seed(7) == 7


def test_jd_seed_is_the_julian_date_in_seconds_modulo_2_32():
    seed = fs.parse_seed("jd", now=J2000_UNIX)
    assert seed == int(2451545.0 * 86400) % 2**32
    assert 0 <= seed < 2**32
    assert fs.parse_seed("JD", now=J2000_UNIX) == seed
    assert fs.parse_seed("jd", now=J2000_UNIX + 1) != seed  # a run a second later gets another seed


@pytest.mark.parametrize("raw", ["abc", "1.5", "-1", str(2**32), "jd2"])
def test_bad_values_raise_with_the_value_in_the_message(raw):
    with pytest.raises(ValueError, match="flux_seed"):
        fs.parse_seed(raw)


def test_env_variable_beats_config():
    assert fs.resolve({"flux_seed": "5"}, env={}) == 5
    assert fs.resolve({"flux_seed": "5"}, env={fs.ENV_VAR: "9"}) == 9
    assert fs.resolve({"flux_seed": "5"}, env={fs.ENV_VAR: ""}) == 5
    assert fs.resolve({}, env={}) == 0


def test_apply_sets_hash_and_rand_seed_and_returns_the_seed():
    env = {"PATH": "/bin", "PERL_HASH_SEED": "99"}
    out, seed = fs.apply(env, {"flux_seed": "77"})
    assert out is env and seed == 77
    assert env["PERL_HASH_SEED"] == "77" and env["PERL_RAND_SEED"] == "77"
    assert env["PATH"] == "/bin"


def test_fixed_seeds_rand_with_one_because_perl_reads_zero_as_unset():
    env, seed = fs.apply({}, {"flux_seed": "0"})
    assert seed == 0 and env["PERL_HASH_SEED"] == "0" and env["PERL_RAND_SEED"] == "1"


def test_apply_jd_uses_now():
    env, seed = fs.apply({}, {"flux_seed": "jd"}, now=J2000_UNIX)
    assert seed == fs.parse_seed("jd", now=J2000_UNIX) and env["PERL_HASH_SEED"] == str(seed)


def test_describe_says_how_to_repeat():
    assert "flux_seed = 4242" in fs.describe(4242)


def test_child_env_reads_config_and_copies_os_environ(monkeypatch):
    monkeypatch.delenv(fs.ENV_VAR, raising=False)
    monkeypatch.setenv("PERL_HASH_SEED", "123")
    env, seed = fs.child_env({"flux_seed": "31"})
    assert seed == 31 and env["PERL_HASH_SEED"] == "31"
    import os

    assert os.environ["PERL_HASH_SEED"] == "123"  # the parent's environment is not touched


def test_config_ini_default_is_fixed():
    from fluxpype.config import configurations

    assert fs.parse_seed(configurations(debug=False).get("flux_seed")) == 0


def test_relax_gives_the_perl_process_the_seed(tmp_path, monkeypatch):
    from fluxpype.scenes import relax as R

    monkeypatch.setenv(fs.ENV_VAR, "4242")
    w = tmp_path / "w.flux"
    w.write_text("x")
    seen = {}

    class Proc:
        pid = 1
        returncode = 0
        stdin = type("In", (), {"write": lambda s, x: None, "close": lambda s: None})()
        stdout = iter(['CYCLE {"cycle":-1,"steps":0,"stiffness":null,"e_simple2":1,"top_z":1,"wall_s":0}\n', "DONE\n"])

        def wait(self):
            return 0

    def popen(args, **kw):
        seen.update(kw)
        return Proc()

    res = R.relax(str(w), cycles=0, popen=popen)
    assert seen["env"]["PERL_HASH_SEED"] == "4242" and seen["env"]["PERL_RAND_SEED"] == "4242"
    assert "flux_seed=4242" in open(res["log"]).read()
