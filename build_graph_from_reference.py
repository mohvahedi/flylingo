"""Build the real MaleCNS v1.0 graph into FlyLingo's shared cache.

Thin driver around the proven reference builder in flm/scripts/prepare_graph.py, pointed at
FlyLingo's own cache paths so the result lands where every other component expects it. The
reference streams the 1 GB edge table in Arrow record batches on purpose, so peak memory
stays low; keep that property.

flm/scripts is not a package, so the reference module is loaded from its file path.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\flylingo")
FLM = ROOT / "flm"

sys.path.insert(0, str(FLM))
sys.path.insert(0, str(FLM / "scripts"))

import flm.paths as paths  # noqa: E402

# Redirect both the source and the output away from flm's own cache.
paths.CACHE = ROOT / "cache"
paths.GRAPH = ROOT / "cache" / "malecns_v1"

import flm.graph as graph_module  # noqa: E402

graph_module.GRAPH = paths.GRAPH


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_graph = _load("prepare_graph", FLM / "scripts" / "prepare_graph.py")
# The reference module bound GRAPH at import time from flm.paths; re-point it at our cache.
prepare_graph.GRAPH = paths.GRAPH

if __name__ == "__main__":
    source = ROOT / "cache" / "source"
    print(f"source: {source}", flush=True)
    print(f"output: {paths.GRAPH}", flush=True)
    prepare_graph.prepare(source)
