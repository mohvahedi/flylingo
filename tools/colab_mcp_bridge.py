"""Drive googlecolab/colab-mcp over stdio and expose it as a local HTTP API.

Hermes picks up new MCP servers only on restart, so instead of touching the running
desktop app this holds one long-lived stdio session to colab-mcp and lets us call its
tools with curl. The session has to stay alive anyway: the Colab websocket proxy dies
with the process.

  GET  /tools                     list tools (re-queried live, so post-connect tools appear)
  GET  /status                    connection state
  POST /call/<tool>               JSON body = arguments; returns the tool result
  GET  /ping                      liveness

Run:  python colab_mcp_bridge.py --port 8791 --log <path>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

COLAB_MCP_ARGS = [
    "--from",
    "git+https://github.com/googlecolab/colab-mcp",
    "colab-mcp",
]


def server_args(source: str) -> list[str]:
    """uvx args for the given source (a git URL or a local patched checkout)."""
    return ["--from", source, "colab-mcp"]

STATE: dict = {
    "ready": False,
    "connected": False,
    "tools": [],
    "last_change": None,
    "log": [],
    "last_error": None,
    "calls": 0,
}


def note(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    STATE["log"].append(line)
    STATE["log"] = STATE["log"][-400:]
    print(line, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    loop: asyncio.AbstractEventLoop = None  # type: ignore
    session: ClientSession = None  # type: ignore

    def log_message(self, *a):  # keep stderr quiet
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _run(self, coro, timeout=900):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/ping":
                return self._send(200, {"ok": True, "ready": STATE["ready"]})
            if path == "/status":
                return self._send(200, {k: STATE[k] for k in
                                        ("ready", "connected", "last_change",
                                         "last_error", "calls")})
            if path == "/tools":
                res = self._run(self.session.list_tools())
                tools = [{
                    "name": t.name,
                    "description": (t.description or "")[:400],
                    # mcp 1.x calls it inputSchema, 2.x input_schema
                    "schema": getattr(t, "inputSchema", None)
                    or getattr(t, "input_schema", None) or {},
                } for t in res.tools]
                STATE["tools"] = [t["name"] for t in tools]
                return self._send(200, {"count": len(tools), "tools": tools})
            if path == "/log":
                return self._send(200, {"log": STATE["log"][-120:]})
            return self._send(404, {"error": "no such path"})
        except Exception as e:  # noqa: BLE001
            STATE["last_error"] = f"{type(e).__name__}: {e}"
            return self._send(500, {"error": STATE["last_error"],
                                    "trace": traceback.format_exc()[-1500:]})

    def do_POST(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) != 2 or parts[0] != "call":
            return self._send(404, {"error": "use POST /call/<tool>"})
        tool = parts[1]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode() if n else "{}"
        try:
            args = json.loads(raw or "{}")
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"bad json: {e}"})
        timeout = float(args.pop("_timeout", 900))
        try:
            STATE["calls"] += 1
            note(f"CALL {tool} {json.dumps(args)[:300]}")
            res = self._run(self.session.call_tool(tool, args), timeout=timeout)
            out = {
                "isError": bool(getattr(res, "isError", False)),
                "structured": getattr(res, "structuredContent", None),
                "content": [getattr(c, "text", str(c)) for c in (res.content or [])],
            }
            # a completed call may have unlocked new tools
            try:
                t = self._run(self.session.list_tools(), timeout=60)
                STATE["tools"] = [x.name for x in t.tools]
            except Exception:  # noqa: BLE001
                pass
            note(f"  -> {json.dumps(out)[:400]}")
            return self._send(200, out)
        except Exception as e:  # noqa: BLE001
            STATE["last_error"] = f"{type(e).__name__}: {e}"
            note(f"  !! {STATE['last_error']}")
            return self._send(500, {"error": STATE["last_error"],
                                    "trace": traceback.format_exc()[-1500:]})


async def amain(port: int, source: str, logdir: str | None) -> None:
    args = server_args(source)
    if logdir:
        args += ["-l", logdir]
    params = StdioServerParameters(command="uvx", args=args)
    note(f"starting colab-mcp ({source}) over stdio ...")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            note("MCP session initialised")

            async def on_message(message):  # notifications from the server
                try:
                    method = getattr(message, "method", None) or getattr(
                        getattr(message, "root", None), "method", None)
                    if method and "list_changed" in str(method):
                        STATE["last_change"] = time.strftime("%H:%M:%S")
                        note("server signalled tools/list_changed")
                except Exception:  # noqa: BLE001
                    pass

            session._received_notification = on_message  # best-effort hook

            Handler.session = session
            Handler.loop = asyncio.get_running_loop()
            STATE["ready"] = True

            httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            note(f"HTTP bridge listening on http://127.0.0.1:{port}")

            while True:
                await asyncio.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--server", default="git+https://github.com/googlecolab/colab-mcp",
                    help="uvx --from source: git URL or a local checkout")
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()
    try:
        asyncio.run(amain(args.port, args.server, args.logdir))
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
