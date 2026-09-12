# viz-fly preview

A three.js fly whose every joint is driven from a 512-value frame of brain activity. No rig
files, no textures, no network: the fly is built from spheres, capsules and boxes in code, so
it renders offline.

## Run it

```bash
cd D:/Projects/flylingo/viz-fly
bun run dev
```

Then open the URL vite prints (http://localhost:5191/ on this machine; vite picks the next
free port if that one is taken).

No backend is needed. The page has no server dependency: it either mounts the stage with no
props at all or feeds itself synthetic frames whose magnitudes match the measured real data.

Production bundle:

```bash
bun run build     # tsc --noEmit && vite build
bun run preview   # serves dist/
```

## What a viewer sees

A dark instrument panel, roughly 1280x800:

- **The fly** in the middle of a stage, facing +Z with its feet on y = 0. Compound eyes, ocelli
  on the vertex, antennae, a proboscis under the head, wings, halteres and six three-segment
  legs. It idles with a breathing abdomen and antenna twitch, walks on a tripod gait, extends
  its proboscis, and grooms its head with a foreleg.
- **Top left, the mode badge**: `INTACT` or `SHUFFLED` or `NO_EDGES` or `RANDOM_GRAPH`, plus
  the "live frames" or "synthetic" marker. This badge is the only thing that distinguishes
  the modes, by design: all three live modes measure nearly the same magnitude (rms 0.105
  intact, 0.108 shuffled, 0.096 random_graph), so the fly is not allowed to look more alive
  for the intact brain.
- **Top right, a sparkline** of the frame rms, scrolling.
- **Bottom left, the dev strip** (only when a frame is supplied and `dev` is on): buttons for
  the five behaviors `idle / walk / groom / proboscis / startle`, an `auto` mode that cycles
  them on a 26 s schedule, a `freeze` toggle, a `speed` multiplier, and `celebrate` / `recoil`
  buttons for the reaction overlays.

The glow carries the activity. Per-frame `abs(activity)` sits at p50 0.051, p90 0.174,
p95 0.274, max 0.717, with about 5 spikes per frame, and the fly's emissive follows that.
Under `no_edges` every one of the 512 values is exactly 0.0 and the fly goes visibly inert:
unlit, no glow, legs planted. On `correct = true` it raises both wings in a flick and hops,
with a warm tint; on `correct = false` it recoils, tucking its legs and slamming both antennae
back, with a cool tint.

## The standalone switcher

The dev entry (`src/main.tsx`) adds one extra panel in the bottom right, outside the stage's
own UI. It is a viewer, not part of the public API:

- `no props` mounts `FlyStage` with no props at all, which is the contract's synthetic idle
  path (an app can mount the stage before its first frame arrives).
- `intact`, `shuffled`, `random_graph` feed synthetic frames at 20 Hz through the same props
  the real app uses.
- `no_edges` feeds 512 exact zeros plus `stateRms 0`, matching the measurement, so the inert
  case can be checked offline.
- `correct=null / true / false` drives the celebrate and recoil reactions.

## Verifying it headlessly

`tools/verify_dev.py` drives the dev server in a real Chromium (swiftshader WebGL) and checks
that the stage mounts, that the canvas actually changes between frames, that each of the five
behaviors draws something different from idle, that `no_edges` is dimmer than `intact`, that
the no-props path animates, and that `correct` drives the reactions.

```bash
python tools/verify_dev.py http://localhost:5191/
```

Last run against `bun run dev` on this machine: no console errors, no page errors, WebGL
context present, and every check passed. canvas brightness `intact` 121.99 vs `no_edges`
42.03 (a drop of 79.96), `celebrate` 123.49 vs `recoil` 99.05, and idle / walk / groom /
proboscis / startle each differ from idle by 4.9% to 56% of pixels. It needs playwright and
Pillow; on this machine the interpreter that has them is
`C:/Users/Lion/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`.

## Contract

The public API is frozen in `../INTERFACES.md`: `FlyStage(props: { activity, stateRms,
activeFraction, correct, reward, mode, width?, height? })`. Everything the standalone page
adds (`dev`, `sparkline`, extra props) is optional and additive. `FlyStage` renders correctly
with no props at all.

## Files

- `src/FlyStage.tsx` public component: canvas, camera, lights, DOM overlays, reaction wiring
- `src/fly/animator.ts` `useAnimator`: frame in, pose plus colour drive out, per rAF
- `src/fly/pose.ts` pure pose math for all five behaviors and the two reactions
- `src/fly/Fly.tsx` the mesh: primitives assembled in code, driven by the pose
- `src/fly/regions.ts` channel layout, mode labels, synthetic frame generator
- `src/components/ControlStrip.tsx`, `src/components/Sparkline.tsx` dev UI
- `src/main.tsx` standalone dev entry
- `tools/verify_dev.py` headless smoke check
