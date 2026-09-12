#!/usr/bin/env python
"""CLI entry point: train the FlyLingo readout and write the checkpoint the API loads.

    python scripts/train_curriculum.py --epochs 30

The service (``brain/api.py``) boots ``PolicyAdapter(in_dim=128, n_actions=4,
hidden=256, seed=0)`` and loads ``brain/runs/curriculum/adapter.npz`` if it
exists, so that exact path is the default output here. Control arms are written
next to it as ``adapter_param_matched.npz`` and ``adapter_shuffled.npz``.

Training uses ``brain/curriculum/es-en.json`` when that file exists. Until the
curriculum workstream lands it, the harness falls back to a built-in synthetic
Spanish-like challenge set and the results JSON says so.

Run this with the shared venv:

    D:/Projects/flylingo/brain/.venv/Scripts/python.exe scripts/train_curriculum.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `import brain...` work no matter where this is invoked from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.learning.train import (  # noqa: E402
    CURRICULUM_JSON,
    DEFAULT_CHECKPOINT,
    DEFAULT_RESULTS,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='train_curriculum.py',
        description='Train the FlyLingo policy adapter on the Spanish curriculum, always '
                    'running the intact connectome arm, a parameter matched control arm, '
                    'and a shuffled graph arm, and write the learning curve as JSON.')
    p.add_argument('--epochs', type=int, default=30,
                   help='passes over the training challenges (default 30)')
    p.add_argument('--curriculum', type=Path, default=CURRICULUM_JSON,
                   help=f'curriculum JSON (default {CURRICULUM_JSON})')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                   help=f'where to write the intact arm checkpoint (default {DEFAULT_CHECKPOINT})')
    p.add_argument('--results', type=Path, default=DEFAULT_RESULTS,
                   help=f'where to write the curve JSON (default {DEFAULT_RESULTS})')
    p.add_argument('--extra-arms', type=str, default='',
                   help='comma list of diagnostic arms to add: no_edges,random_graph')
    p.add_argument('--eval-fraction', type=float, default=0.25,
                   help='fraction of challenges held out for the eval curve (default 0.25)')
    p.add_argument('--seed', type=int, default=0,
                   help='adapter seed; the control arms reuse it so init is matched')
    p.add_argument('--hidden', type=int, default=256,
                   help='adapter hidden width; must stay 256 to match brain/api.py')
    p.add_argument('--lr', type=float, default=0.05, help='policy learning rate')
    p.add_argument('--gate-lr', type=float, default=0.05, help='dopamine gated learning rate')
    p.add_argument('--shuffle-epochs', action='store_true',
                   help='shuffle the training order per epoch (same order in every arm)')
    p.add_argument('--standin', action='store_true',
                   help='force the labelled stand-in reservoir even if brain/reservoir.py exists')
    p.add_argument('--no-save', action='store_true', help='do not write checkpoints')
    p.add_argument('--json', action='store_true', help='print the results JSON to stdout')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    extra = tuple(a.strip() for a in args.extra_arms.split(',') if a.strip())
    adapters = ['intact', 'param_matched', 'shuffled']
    print(f'FlyLingo training: arms={adapters} extra={list(extra)} epochs={args.epochs}')
    if not args.curriculum.exists():
        print(f'NOTE: {args.curriculum} is not on disk yet; the harness will use the '
              f'built-in synthetic Spanish-like set and label the results as synthetic.')
    results = train(
        epochs=args.epochs,
        extra_arms=extra,
        curriculum=args.curriculum,
        results_path=args.results,
        checkpoint_path=args.checkpoint,
        eval_fraction=args.eval_fraction,
        adapter_seed=args.seed,
        hidden=args.hidden,
        lr=args.lr,
        gate_lr=args.gate_lr,
        shuffle_epochs=args.shuffle_epochs,
        force_standin=args.standin,
        save_checkpoints=not args.no_save,
    )
    print()
    print('summary')
    print(f'  reservoir   : {results["reservoir"]["source"]} '
          f'(standin={results["reservoir"]["is_standin"]})')
    print(f'  task        : {results["task"]["source"]} '
          f'{results["task"]["n_train"]} train / {results["task"]["n_eval"]} eval')
    for name, arm in results['arms'].items():
        if arm['ran']:
            print(f'  {name:<14}: final eval {arm["final"]["eval_accuracy"]:.3f} '
                  f'(best {arm["final"]["best_eval_accuracy"]:.3f}) '
                  f'params {arm["params"]} gated {arm["gated_params"]} '
                  f'mode {arm["reservoir_mode_used"]}')
        else:
            print(f'  {name:<14}: DID NOT RUN ({arm["skipped_reason"]})')
    print(f'  verdict     : {results["comparison"].get("verdict")}')
    if results['reservoir']['is_standin']:
        print()
        print('WARNING: ' + results['reservoir']['warning'])
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
