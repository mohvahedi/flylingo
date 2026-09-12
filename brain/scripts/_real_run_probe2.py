"""Throwaway part 2: timing + modes of the real reservoir."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/Projects/flylingo")
sys.path.insert(0, str(ROOT / 'brain'))
from brain.learning import train as T  # noqa: E402

ts = T.load_task(eval_fraction=0.25, seed=T.ADAPTER_SEED, n_actions=T.N_ACTIONS)
res, meta = T.load_reservoir(embedding_dim=T.EMBED_DIM, dims=T.DIMS,
                             seed=T.RESERVOIR_SEED, force_standin=False)
print("class", type(res).__name__, "mode", getattr(res, "mode", None), "standin", meta["is_standin"])
E = np.stack([T.encode_challenge(c, T.EMBED_DIM) for c in ts.challenges])
print("E", E.shape, "unit rms?", float(np.sqrt((E ** 2).sum(1).mean())))
t0 = time.time()
F = res.sequence(E, "intact")
dt = (time.time() - t0) / len(E)
tel = getattr(res, "telemetry", lambda: {})()
print("97-step sequence: total %.2fs  %.1f ms/step -> %.1f Hz" % (time.time() - t0, dt * 1e3, 1 / dt))
print("F", F.shape, "state_rms", float(np.sqrt((F ** 2).mean())),
      "per-step rms", float(np.sqrt((F ** 2).sum(1).mean())))
print("telemetry keys", sorted(tel.keys()))
print({k: v for k, v in tel.items() if not isinstance(v, list)})
ts_tel = {}
for m in ("intact", "shuffled", "random_graph", "no_edges"):
    try:
        res.set_mode(m)
        a = res.sequence(E[:10], m)
        ts_tel[m] = ("OK", res.mode, float(np.sqrt((a ** 2).mean())))
    except Exception as exc:
        ts_tel[m] = ("REFUSED", repr(exc)[:160])
print("modes:", json.dumps(ts_tel, default=str))
