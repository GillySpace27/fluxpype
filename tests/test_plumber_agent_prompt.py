"""The copy-paste prompt that connects an AI agent to the Plumber MCP server."""

import json
import sys

import pytest

from fluxpype.plumber import agent_prompt


def test_build_names_this_python_and_home(tmp_path):
    out = agent_prompt.build(python=sys.executable, home=tmp_path, demo=False)
    assert out["name"] == "fluxpype"
    assert out["args"] == ["-m", "fluxpype.plumber.mcp_server", "--home", str(tmp_path.resolve())]
    assert out["claude_code"].startswith("claude mcp add --scope user fluxpype -- ")
    assert json.loads(out["json_config"])["mcpServers"]["fluxpype"]["command"] == sys.executable
    assert sys.executable in out["prompt"] and str(tmp_path.resolve()) in out["prompt"]
    assert "do not submit, cancel or retry" in out["prompt"]
    assert "—" not in out["prompt"]


def test_demo_gets_its_own_name_and_flag(tmp_path):
    out = agent_prompt.build(python=sys.executable, home=tmp_path, demo=True, check=False)
    assert out["name"] == "fluxpype-demo" and out["args"][-1] == "--demo"
    assert "simulated demo" in out["prompt"]


def test_missing_sdk_adds_install_step(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_prompt, "probe", lambda py, cwd: {"fluxpype": True, "mcp": False})
    out = agent_prompt.build(python="/opt/venv/bin/python", home=tmp_path)
    assert "pip install" in out["prompt"] and out["install_command"].startswith("/opt/venv/bin/python -m pip")
    monkeypatch.setattr(agent_prompt, "probe", lambda py, cwd: {"fluxpype": True, "mcp": True})
    assert "pip install" not in agent_prompt.build(python="/opt/venv/bin/python", home=tmp_path)["prompt"]


def test_path_with_spaces_is_quoted_for_the_shell(tmp_path):
    home = tmp_path / "my home"
    out = agent_prompt.build(python="/Users/g/a b/python", home=home, check=False)
    assert "'/Users/g/a b/python'" in out["claude_code"] and f"'{home.resolve()}'" in out["claude_code"]


def test_probe_reports_unknown_for_a_missing_interpreter(tmp_path):
    assert agent_prompt.probe(str(tmp_path / "nope"), str(tmp_path)) == {"fluxpype": None, "mcp": None}


def test_endpoint_returns_prompt_without_token(make_scheduler):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from fluxpype.plumber.server import create_app

    token = "secret-token-xyz"
    sched = make_scheduler()
    with TestClient(create_app(sched, token, demo=True, run_loop=False), base_url="http://localhost") as c:
        assert c.get("/api/agent-prompt").status_code == 401
        body = c.get("/api/agent-prompt", headers={"X-Flux-Token": token}).json()
    assert body["name"] == "fluxpype-demo" and body["home"] == str(sched.home.resolve())
    assert token not in json.dumps(body)


def test_mcp_cli_print_prompt(tmp_path, capsys):
    pytest.importorskip("httpx")
    pytest.importorskip("mcp")
    from fluxpype.plumber import mcp_server

    assert mcp_server.main(["--print-prompt", "--home", str(tmp_path)]) == 0
    assert "fluxpype.plumber.mcp_server" in capsys.readouterr().out
    assert mcp_server.main(["--print-prompt", "--url", "http://localhost:1", "--token", "x"]) == 2
