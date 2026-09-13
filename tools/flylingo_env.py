"""Resolve the ports the demo actually runs on.

Why this exists: on this host Windows randomly reserves large TCP ranges for Hyper-V on
every boot (`netsh interface ipv4 show excludedportrange protocol=tcp`). After one reboot the
ranges covered 1663-3179 continuously, so the app could no longer bind 3100 and died with
`EACCES: permission denied 0.0.0.0:3100`. Nothing in the code was wrong; the port had simply
become unusable. Hardcoding a port therefore breaks the harness at random.

Resolution order, so the tools work whether or not you pass anything:

  1. an explicit override (env `FLYLINGO_APP` / `FLYLINGO_API`, or a CLI arg)
  2. the port recorded in `tools/.ports.json` (written when the servers are started)
  3. a default, if it happens to be bindable

`find_free_port` is provided for the launcher so a port can be chosen instead of guessed.

Usage from a tool:

    from flylingo_env import APP_BASE, API_BASE, app_url

    app_url("/lesson/fly")   ->  "http://127.0.0.1:3300/lesson/fly"

Run directly to print what is currently resolved:

    python tools/flylingo_env.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORTS_FILE = Path(__file__).resolve().parent / ".ports.json"

DEFAULT_APP_PORT = 3300
DEFAULT_API_PORT = 8770


def _bindable(port: int, host: str = "0.0.0.0") -> bool:
    """True if we could bind this port right now."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def find_free_port(preferred: int = DEFAULT_APP_PORT, tries: int = 60) -> int:
    """First bindable port at or above `preferred`.

    Deliberately not restricted to a fixed list: the excluded ranges move, so the only
    reliable test is an actual bind attempt.
    """
    for port in range(preferred, preferred + tries):
        if _bindable(port):
            return port
    raise RuntimeError(f"no free port in {preferred}..{preferred + tries}")


def _from_file(key: str) -> int | None:
    try:
        data = json.loads(PORTS_FILE.read_text(encoding="utf-8"))
        value = data.get(key)
        return int(value) if value is not None else None
    except Exception:
        return None


def write_ports(app: int, api: int) -> Path:
    """Record the ports the servers were actually started on."""
    PORTS_FILE.write_text(
        json.dumps({"app": app, "api": api}, indent=2) + "\n", encoding="utf-8"
    )
    return PORTS_FILE


def app_port(cli_arg: str | None = None) -> int:
    if cli_arg:
        return int(str(cli_arg).rstrip("/").rsplit(":", 1)[-1])
    env = os.environ.get("FLYLINGO_APP")
    if env:
        return int(str(env).rstrip("/").rsplit(":", 1)[-1])
    recorded = _from_file("app")
    if recorded:
        return recorded
    return DEFAULT_APP_PORT


def api_port(cli_arg: str | None = None) -> int:
    if cli_arg:
        return int(str(cli_arg).rstrip("/").rsplit(":", 1)[-1])
    env = os.environ.get("FLYLINGO_API")
    if env:
        return int(str(env).rstrip("/").rsplit(":", 1)[-1])
    recorded = _from_file("api")
    if recorded:
        return recorded
    return DEFAULT_API_PORT


# Resolved once at import for the common case. Tools that take a CLI arg should call the
# functions above with `sys.argv[1]` instead.
APP_PORT = app_port()
API_PORT = api_port()
APP_BASE = f"http://127.0.0.1:{APP_PORT}"
API_BASE = f"http://127.0.0.1:{API_PORT}"


def app_url(path: str = "/") -> str:
    if path.startswith("http"):
        return path
    if path.startswith(":"):  # caller passed a port
        path = "/" + path.split(":", 1)[1].lstrip("/")
    return APP_BASE + ("" if path.startswith("/") else "/") + path


def api_url(path: str = "/") -> str:
    if path.startswith("http"):
        return path
    return API_BASE + ("" if path.startswith("/") else "/") + path


if __name__ == "__main__":
    print(f"ports file      {PORTS_FILE}  exists={PORTS_FILE.exists()}")
    print(f"app  -> {APP_BASE}   bindable_now={_bindable(APP_PORT)}  in_use={not _bindable(APP_PORT)}")
    print(f"api  -> {API_BASE}")
    print(f"first free app port from {DEFAULT_APP_PORT}: {find_free_port()}")
    if len(sys.argv) > 1:
        print(f"cli override -> app {app_port(sys.argv[1])}")
