# viz-brain preview

Runnable demo for the BrainCloud component: a real Vite + React + TypeScript app
that draws the MaleCNS cloud from `public/data`, feeds it live frames from the
brain service, and falls back to the synthetic idle animation when the socket is
not there. All numbers below were measured on this machine, not estimated.

## Run it

Install (already done in this workspace, `node_modules` is present):

```
cd D:/Projects/flylingo/viz-brain
bun install
```

Dev server:

```
cd D:/Projects/flylingo/viz-brain
bun run dev
```

Then open <http://127.0.0.1:5180/>. The dev script is
`vite --port 5180 --strictPort`, so the port is fixed and never silently moves.

Production build and preview:

```
cd D:/Projects/flylingo/viz-brain
bun run build      # tsc -b && vite build, exits 0, writes dist/
bun run preview    # serves dist/ on http://127.0.0.1:4180/
```

`bun run build` output: 68 modules, `dist/index.html` 0.34 kB,
`dist/assets/index-*.js` 1014 kB (282 kB gzip, three.js is most of it),
built in about 1.9 s. `public/data` is copied to `dist/data` by the build.

## What a viewer sees

A single dark page, three regions:

- Header: `FlyLingo BrainCloud harness`, a mode badge (`MODE INTACT`,
  `MODE SHUFFLED`, `MODE NO_EDGES`, `MODE RANDOM_GRAPH`, or `SYNTHETIC IDLE`
  when no socket), a source pill (`live ws://127.0.0.1:8770/stream` or
  `socket disabled, idle animation`), a layout badge (`anatomical soma
  coordinates` or `procedural layout`), and the drive toggle (`subset` / `full`).
  On the right, a live FPS readout: the current rate, average frame interval and
  worst frame interval, all from `src/metrics/fps.ts`.
- Middle left: the BrainCloud canvas with the 166,700 point cloud. Overlaid text
  states which layout is in use, the annotation accuracy numbers, the auto range
  rule, and the mode caveat. Bottom right of the panel, the detail table: mode,
  drive mode, points, live slots, resolved ids and mapping basis, reference
  value, peak `|state|`, spike count, bytes uploaded per frame, renderer string,
  socket step and challenge id.
- Middle right: the raster strip, 512 rows by up to 240 frame columns, 12 s of
  history, bright marks where a spike landed. Caption shows the frame count and
  the active source.

Colors and camera: the cloud is drawn with the existing shader in
`src/shaders/points.ts`; idle frames animate and live frames replace them slot
by slot in the 512 live slots.

## The socket

- Endpoint: `ws://127.0.0.1:8770/stream` (the contract's port is 8770).
- The client is `src/live/socket.ts`. It decodes the frozen `BrainFrame`
  (snake_case keys, `state` in -1..1, `spikes` as indices into `state`,
  `sampled_ids` as strings), rejects frames without a state array, drops spikes
  that are not a valid index into state, and reports how many it dropped.
- Reconnect is automatic: 0.5 s, 1 s, 2 s, then 4 s, forever, until unmounted.
  A socket that never opens is not an error state, the page just stays on the
  idle animation and the source pill says so.
- `src/hooks/useBrainStream.ts` owns the connection, keeps the frame ring for
  the raster strip, and runs the synthetic idle driver at 20 Hz from
  `src/hooks/useSyntheticFrame.ts` whenever the socket is not live. Switching
  from synthetic to live clears the ring so the strip does not mix sources
  without a mark.

Verified live against the running service: 512 of 512 `sampled_ids` resolved to
positions in the real layout, `mode intact`, `synthetic false`, reference value
0.2414 (95th percentile of the frame), peak `|state|` 0.7924, 4 spikes in the
frame.

## Query parameters

Used by the measurement script and by hand:

| param | effect |
| --- | --- |
| `drive=subset` (default) | write only the 512 sampled slots per frame (10 KB upload) |
| `drive=full` | write all 166,700 slots per frame (651 KB upload) |
| `stream=off` | never open the socket, stay on the synthetic idle path |
| `stream=ws://host:port/path` | override the stream URL |

## Measured frame rate

Tool: `scripts/measure-fps.mjs`, run as `bun run fps --serve` (or with
`--url` against a server you started yourself, `--modes subset,full`,
`--seconds`, `--warmup`, `--uncapped`, `--json`). It launches the installed
Chrome headless over the DevTools protocol, samples `window.__bc` every 500 ms
and reports what the page rendered while live frames were arriving. No npm
dependency beyond the browser.

Environment: this machine's Google Chrome, headless, 1600x900,
`ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)`
i.e. software rasterization, no GPU in the headless session. Cloud size 166,700
points, 512 live slots, live socket at about 20 Hz.

| scenario | fps mean | fps median | sample min / max | avg frame | worst frame |
| --- | --- | --- | --- | --- | --- |
| preview build, drive=subset, 4 runs of 10 s | 23.71 / 24.67 / 24.85 / 27.01 | 21.35 / 23.16 / 23.21 / 23.79 | 11.31 / 49.37 | 79.5 to 88.4 ms | 788 to 1023 ms |
| preview build, drive=full, 4 runs of 10 s | 24.11 / 24.64 / 25.38 / 26.72 | 20.40 / 21.75 / 22.32 / 23.77 | 11.90 / 49.09 | 79.8 to 84.1 ms | 711 to 916 ms |
| preview build, `--uncapped` (vsync off), subset / full | 25.13 / 25.28 | 24.80 / 22.44 | 12.43 / 47.48 | 78.6 to 80.5 ms | 738 to 867 ms |
| preview build, subset / full, 8 s | 27.53 / 27.46 | 25.87 / 24.63 | 14.17 / 49.75 | 67.7 to 70.6 ms | 668 to 830 ms |
| preview build, subset / full, 6 s | 30.70 / 30.47 and 30.02 / 30.19 | 29.75 / 30.79 | 17.19 / 49.32 | 52.7 to 58.2 ms | 496 to 646 ms |
| dev server, drive=subset, 8 s | 28.72 | 27.72 | 14.65 / 48.21 | 68.3 ms | 680 ms |
| dev server, drive=full, 8 s | 28.36 | 25.82 | 14.81 / 50.43 | 67.5 ms | 633 ms |

Headline: about 25 to 30 fps mean in this environment, with the mean sliding with
the window length (23.7 over 10 s, 27.5 over 8 s, 30.3 over 6 s) because a longer
window catches more of the multi hundred millisecond stalls. Worst frames run 496
to 1023 ms while SwiftShader stalls. The uncapped run lands in the same place as
the vsync run, so this is not a vsync ceiling: the page is CPU bound in software
point rasterization. Treat about 27 fps as the representative number for a viewer
on this box, and the per sample floor (11 to 19 fps) as what happens during a
stall.

The same runs collected the page's own console and log output: zero errors and
zero warnings in both drive modes with the live socket and the anatomical layout.
The only message ever seen was the browser's automatic `/favicon.ico` request
returning 404, which `index.html` now silences with an inline empty icon.

The UI always draws the full 166,700 point cloud. The 512 number is the live
slot count, not a point count: there is no 512 point render path in the harness.
`drive` only changes how many slots are uploaded per frame (10 KB versus 651 KB),
and the two modes measure within noise of each other, which says the per frame
upload is not the bottleneck at this size. If you need a higher number, the
lever is the renderer (a real GPU instead of SwiftShader) or the point count,
not the drive mode.

## Data accuracy claims in the UI

The overlay states the measured facts and the component does not claim more:

- `MaleCNS v1.0: 139662 of 166700 neurons carry a measured soma annotation
  (somaLocation); 27038 are placed at their centroid of (class, in-degree
  decile) group because no soma was annotated.`
- Layout badge: `anatomical soma coordinates` when
  `public/data/soma_positions.f32` loaded, `procedural layout` when it did not.

When the data cannot be loaded, the app builds the procedural layout from
`buildProceduralLayout` and says so out loud: amber banner, badge reads
`procedural layout`, overlay reads `coordinates: procedural fallback, not
anatomy`, and the reason (the fetch error) is printed. Verified by renaming
`dist/data` and re-running the page.

## Fixes made while finishing the harness

- Added `src/main.tsx` and `src/App.tsx` plus `src/live/socket.ts`,
  `src/hooks/useBrainStream.ts`, `src/ui/ModeBadge.tsx`,
  `src/ui/FpsReadout.tsx`, `src/ui/MetricsPanel.tsx`,
  `src/ui/useMetricsSnapshot.ts`.
- Fixed `src/data/loadNeuronData.ts`: `DATA_BASE` was
  `new URL('./data/', import.meta.url)`, which resolves to `src/data/data` in
  dev and therefore never loaded. It now resolves against the document, so
  `/data/*` is fetched from `public/data` in dev and in a build.
- Added `scripts/measure-fps.mjs` (the `bun run fps` script that package.json
  already advertised but did not have). It shuts the browser down through
  `Browser.close` and kills process trees, so a run does not leave a headless
  browser on the devtools port or a preview server on 4180.
- `index.html` gained one line, `<link rel="icon" href="data:," />`, which stops
  the browser's automatic `/favicon.ico` request from showing up as a console
  404. The module script still points at `/src/main.tsx`, which now exists.

The lead's earlier type fixes are untouched and still in place: `BrainCloud.tsx`
does not destructure `onMetrics` (the prop stays in the type), the synthetic
branch returns `{ ...syntheticFrame(tMs / 1000), sampledIds: [] }`, and
`layout/index.ts` keeps `void seed;` and the removed `binCount`.

## Known limits

- The FPS figures come from a software renderer in a headless session on a
  shared CPU. They are a floor for this box, not a hardware benchmark.
- The raster strip keeps 240 frames (about 12 s at 20 Hz).
- Live frames are at about 20 Hz, so slot updates are not a per display frame
  signal; the cloud interpolation between frames is what the FPS number is
  measuring.
