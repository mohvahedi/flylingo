"""Read-only inventory: curriculum path/structure, trainer constants, run dirs.

Prints a compact report. Redirect stdout to a file and read it back; terminal
capture on this host is unreliable for long output.
"""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = []


def say(*parts):
    OUT.append(" ".join(str(p) for p in parts))


say("ROOT", ROOT)
say("cwd", Path.cwd())

candidates = [
    ROOT / "brain" / "curriculum" / "es-en.json",
    ROOT / "brain" / "brain" / "curriculum" / "es-en.json",
    ROOT / "brain" / "brain" / "brain" / "curriculum" / "es-en.json",
]
for p in candidates:
    say("curriculum candidate exists=", p.exists(), p)

for d in [ROOT, ROOT / "brain", ROOT / "brain" / "brain", ROOT / "brain" / "runs",
          ROOT / "brain" / "tests", ROOT / "brain" / "scripts"]:
    if d.is_dir():
        say("DIR", d, sorted(x.name for x in d.iterdir())[:40])
    else:
        say("MISSING DIR", d)

try:
    from brain.learning import train as T

    for name in ("DIMS", "EMBED_DIM", "HIDDEN", "N_ACTIONS", "ADAPTER_SEED",
                 "RESERVOIR_SEED", "DEFAULT_CHECKPOINT", "DEFAULT_RESULTS",
                 "RUNS_DIR", "PARAMETER_NAMES", "ARM_SPECS", "EXTRA_ARM_SPECS",
                 "DATA_DEFAULT", "CURRICULUM", "TASK_JSON", "STANDIN_WARNING"):
        if hasattr(T, name):
            say("CONST", name, "=", repr(getattr(T, name)))
    say("--- load_task source ---")
    say(inspect.getsource(T.load_task))
    say("--- features_for source ---")
    say(inspect.getsource(T.features_for))
    say("--- make_adapters source ---")
    say(inspect.getsource(T.make_adapters))
    say("--- load_reservoir source ---")
    say(inspect.getsource(T.load_reservoir))
except Exception as exc:  # pragma: no cover - diagnostics
    say("IMPORT train FAILED", type(exc).__name__, exc)

try:
    from brain.encoders import encoder_fingerprint, EMBED_DIM

    say("fingerprint", encoder_fingerprint(), "EMBED_DIM", EMBED_DIM)
except Exception as exc:
    say("IMPORT encoders FAILED", type(exc).__name__, exc)

p = ROOT / "brain" / "runs" / "curriculum" / "results.json"
if p.exists():
    blob = json.loads(p.read_text())
    say("prior results task:", json.dumps(blob.get("task"), indent=1)[:1200])
    say("prior results comparison:", json.dumps(blob.get("comparison"), indent=1)[:1200])
    for arm, a in (blob.get("arms") or {}).items():
        say("prior arm", arm, "final", json.dumps(a.get("final")))

text = "\n".join(OUT)
Path(sys.argv[1] if len(sys.argv) > 1 else "inv.txt").write_text(text, encoding="utf-8")
print("WROTE", len(text), "chars")
