# FLUXpype

## Preamble
This package acts as a wrapper around the Field Line Universal relaXer (FLUX), a solar magnetic field model. This means that it requires the FLUX perl/C software located on Github.

## Installation

FLUX has two halves: the **core** (a C library with Perl/PDL bindings, in
[fluxon-mhd](https://github.com/lowderchris/fluxon-mhd)) and this **Python wrapper**.

1. **Build the core** by following the guide for your platform in
   [fluxon-mhd/doc](https://github.com/lowderchris/fluxon-mhd/tree/main/doc).
   It is done when, in a *new* terminal, `perl -MPDL -MFlux -e1` prints nothing and returns.
2. **Install fluxpype** (Python 3.10 to 3.12):

   ```sh
   git clone https://github.com/GillySpace27/fluxpype.git
   cd fluxpype
   ./install.sh
   ```

   This creates `./.venv`, installs fluxpype into it, and runs `flux-doctor`,
   which checks Python, Perl, PDL, Flux, gnuplot and the config, and prints
   the fix for anything missing. Run `flux-doctor` again any time.
3. **Set your JSOC email** in `fluxpype/config.ini` (`jsoc_email`) or as
   `JSOC_EMAIL`. JSOC only sends data to registered addresses
   ([register here](http://jsoc.stanford.edu/ajax/register_email.html)).
4. **First run** (one rotation, 100 fluxons, about 10 minutes):

   ```sh
   source .venv/bin/activate
   flux-config-run --config quickstart
   ```

   `flux-config-run` exits nonzero if any job fails, and prints which ones.

## Configuration
Once the code is all installed correctly, you'll be able to edit the configuration file and run it in the following manner.

Editing the configuration file by running the flux-config-edit command is one of the primary ways to interface with the code. The file can also be edited directly. The second line of the config file (`config_name`) selects the default profile; `flux-config-run --config NAME` overrides it for one run. Any settings not redefined in that profile will use the default settings from the first block ("DEFAULT").

To see the details of the current job configuration:
> flux-config-view

To edit the details of the configuration, modify config.ini. You can find that by typing:
> flux-config-edit

## Running
The pipe can be invoked in four ways.

> \% flux-config-run

> \% python3 fluxpype/config_runner.py

> \% python3 fluxpype

> \% python3 \
>\>\>\> from fluxpype.config_runner import run \
>\>\>\> run()


## Fluxpype Plumber

A local web GUI for running and watching the pipeline: choose a profile, override settings for one run, queue tasks, follow stages and relaxation live, browse plots and logs, and clean up cached files.

```bash
pip install -e ".[plumber]"
flux-plumb              # opens http://localhost:8765 (a one-time token URL is printed)
flux-plumb --demo       # simulated pipeline, no FLUX/PDL needed
flux-plumb --stop       # stop the Plumber for this state directory (running jobs keep running)
```

Only one Plumber runs per state directory (a lock file enforces it); a second `flux-plumb` prints where the first is and exits with status 3. `--headless` runs without a browser or stdout output and logs to `<home>/server.log`.

Plumber binds to 127.0.0.1 only. On a remote machine use an SSH tunnel (`ssh -L 8765:localhost:8765 host`). It never writes `config.ini`; per-run overrides are applied to a snapshot copy. The terminal behavior of `flux-config-run` is unchanged unless the Plumber environment variables are set. See `docs/gui/PLAN.md` and `docs/gui/EVENTS.md`.

Frontend sources live in `fluxpype/plumber/web` (React, TypeScript, Vite, Tailwind). The built bundle in `dist/` is committed, so users do not need Node. To rebuild: `cd fluxpype/plumber/web && npm ci && npm run build`.

### MCP server

`flux-plumb-mcp` lets an AI assistant (Claude Code, Claude Desktop, any MCP client) look at and drive Plumber: submit jobs, wait for stages, read logs and relaxation curves, look at plots. It is a client of Plumber's local API, speaks MCP over stdio, and finds (or, if none is running, starts) the Plumber for your state directory. Tools, safety notes and the autostart lifetime are in [docs/mcp/TOOLS.md](docs/mcp/TOOLS.md).

The quickest way to connect an agent: open Plumber, click **Connect agent** in the top bar (also under Settings), and paste the copied prompt into your agent. The prompt carries this Plumber's exact Python and state directory, checks whether the MCP SDK is installed, and asks the agent to register the server and confirm the connection without starting any jobs. It contains no token. The same prompt is available from a terminal:

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/python -m fluxpype.plumber.mcp_server --print-prompt
```

To set it up by hand instead, install the extra into the same environment you use for Plumber:

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/pip install -e ".[mcp]"
```

Claude Code:

```bash
cd ~/vscode/fluxons-plumber/fluxpype
claude mcp add fluxpype -- "$PWD/.venv-plumber/bin/python" -m fluxpype.plumber.mcp_server
```

Claude Desktop (`claude_desktop_config.json`; use the absolute path of the venv's Python):

```json
{
  "mcpServers": {
    "fluxpype": {
      "command": "/home/you/vscode/fluxons-plumber/fluxpype/.venv-plumber/bin/python",
      "args": ["-m", "fluxpype.plumber.mcp_server"],
      "cwd": "/home/you/vscode/fluxons-plumber/fluxpype"
    }
  }
}
```

A Plumber started this way keeps running after the assistant exits, because its jobs must. Stop it with:

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/flux-plumb --stop
```

Running jobs are detached and keep going; the next Plumber picks them up. Options: `--home PATH`, `--url` with `--token-file` for an already running local Plumber, `--no-autostart`, and `--demo` (needs `--home`) for a simulated pipeline.

![Overview](docs/gui/screenshots/overview.png)

## Contributing

This project welcomes contributions and suggestions. For details, visit the repository's [Contributor License Agreement (CLA)](https://cla.opensource.microsoft.com) and [Code of Conduct](https://opensource.microsoft.com/codeofconduct/) pages.
