"""Throwaway: inspect the real curriculum and the real reservoir before training."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"D:/Projects/flylingo")
sys.path.insert(0, str(ROOT / 'brain'))
from brain.learning import train as T  # noqa: E402

for c in T.CURRICULUM_CANDIDATES:
    print("candidate", c, c.exists())
p = next((q for q in T.CURRICULUM_CANDIDATES if q.exists()), None)
print("chosen:", p)
raw = json.loads(p.read_text(encoding="utf-8"))
print("top keys:", sorted(raw.keys()))
units = raw.get("units", [])
print("units", len(units), "lessons", sum(len(u.get("lessons", [])) for u in units))
ch = [c for u in units for l in u.get("lessons", []) for c in l.get("challenges", [])]
print("challenges", len(ch))
if ch:
    print("keys of first:", sorted(ch[0].keys()))
    print("sample:", json.dumps({k: v for k, v in ch[0].items()}, ensure_ascii=False)[:400])
bad = [(_validate := T._validate(c), c.get("id")) for c in ch]
bad = [(r, i) for r, i in bad if r]
print("invalid:", len(bad), bad[:5])
from collections import Counter  # noqa: E402
print("n_options:", Counter(len(c.get("options", [])) for c in ch))
print("difficulty:", Counter(int(c.get("difficulty", 1)) for c in ch))
print("correctIndex:", Counter(int(c.get("correctIndex", -1)) for c in ch))
print("types:", Counter(c.get("type") for c in ch))

ts = T.load_task(eval_fraction=0.25, seed=T.ADAPTER_SEED, n_actions=T.N_ACTIONS)
print("TaskSet:", json.dumps(ts.summary(), ensure_ascii=False)[:600])

print("--- reservoir")
t0 = time.time()
res, meta = T.load_reservoir(embedding_dim=T.EMBED_DIM, dims=T.DIMS,
                             seed=T.RESERVOIR_SEED, force_standin=False)
print("load %.2fs" % (time.time() - t0), json.dumps(meta, default=str)[:600])
print("class:", type(res).__name__, "mode:", getattr(res, "mode", None))
print("telemetry:", json.dumps(getattr(res, "telemetry", lambda: {})(), default=str)[:1200])
e = T.encode_challenge(ts.challenges[0], T.EMBED_DIM)
t0 = time.time()
f = res.sequence(np.stack([e] * 5), "intact")
dt = (time.time() - t0) / 5
print("5-step sequence %.1f ms/step -> %.1f Hz" % (dt * 1e3, 1 / dt))
print("feats shape", f.shape, "state_rms", float(np.sqrt((f ** 2).mean())))
for m in ("shuffled", "random_graph", "no_edges"):
    try:
        res.set_mode(m)
        print("mode", m, "OK ->", res.mode)
    except Exception as exc:
        print("mode", m, "REFUSED", repr(exc)[:200])
