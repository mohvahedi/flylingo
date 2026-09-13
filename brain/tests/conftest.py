"""Shared test isolation.

`brain.api` now persists the demo's plastic brain to brain/runs/plastic_brain/demo.npz on
shutdown, and every `TestClient(app)` context manager fires that shutdown. Two problems if the
real path is used:

  * the suite would write into the repo's runtime directory while testing, and
  * worse, one test's trained brain would be picked up by the next test that expects a fresh one,
    so a result would depend on test order.

The path is redirected per test instead, the same way HERMES_HOME is redirected per test in
Hermes itself. Tests that care about the location get it as a fixture argument.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_demo_checkpoint(tmp_path, monkeypatch):
    """Point the demo's plastic-brain checkpoint at a per-test temporary directory."""
    try:
        import brain.api as api
    except Exception:  # noqa: BLE001 - tests that do not touch the API need no redirection
        return None
    target = tmp_path / "demo.npz"
    monkeypatch.setattr(api, "DEMO_PLASTIC_CHECKPOINT", target, raising=False)
    return target
