"""Golden master for the MCP tool surface: tool names, argument names and annotations of both server copies.

    python tests/mcp_tools_golden.py --update    rewrite tests/golden/mcp_tools.golden

"agent" is the copy AI agents get; "ask" is the Ask bar's copy (FLUX_MCP_CLIENT=ask), whose writes only propose.
test_plumber_mcp.py fails when either copy gains, loses or renames a tool or an argument, or changes an annotation.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden", "mcp_tools.golden")


def surface(server):
    import anyio
    from mcp import Client

    async def go():
        async with Client(server) as c:
            return (await c.list_tools()).tools

    out = {}
    for t in anyio.run(go):
        a = t.annotations
        out[t.name] = {
            "args": sorted((t.input_schema or {}).get("properties", {})),
            "annotations": {
                k: getattr(a, k)
                for k in ("read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint")
                if a is not None and getattr(a, k) is not None
            },
        }
    return dict(sorted(out.items()))


def table():
    sys.path.insert(0, ROOT)
    from fluxpype.plumber.mcp_server import PlumberClient, build_server

    client = PlumberClient(base_url="http://localhost:9", token="x")  # never called: list_tools needs no Plumber
    return {"agent": surface(build_server(client)), "ask": surface(build_server(client, propose=True))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    rows = table()
    if a.update:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(rows, f, indent=1, sort_keys=True)
            f.write("\n")
        print("wrote", GOLDEN)
    else:
        print(json.dumps(rows, indent=1, sort_keys=True))
