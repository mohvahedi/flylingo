# Connectome panel honesty audit, amber accent + dead-brain control

Measured with `tools/hud_shots.py` (writes `artifacts/hud_shots.json`, gitignored) against
`bun run dev --port 4180`, plus two standalone Playwright probes. Every number below is read
from a file, not estimated.

## Defect 1, amber fresh-spike accent

`SPIKE_FLASH_SECONDS = 0.16` decays in wall clock, and `SPIKE_HOLD_FRAMES = 12` pins a reported
spike at full strength for 12 **rendered frames**. At the measured 28 to 29 fps that hold is only
0.41 s, and it shrinks as the renderer gets faster, legibility was inversely frame-rate dependent.
The flash is also refreshed whenever the frame lists a spike, so it never depended on a spike-key
change.

Post-fix, three harness runs (source of truth), `warm_px` / `hot_px` on the six cloud-live captures:

| run | warm_px | hot_px | no_edges lit | ratio | amber flag |
|---|---|---|---|---|---|
| A (fps 18) | 15 to 42 | 6 | 2.91% | 9.81 | true |
| C (fps 28.4) | 126, 41, 225, 19, 152, 15 | 7, 16, 55, 12, 24, 6 | 2.91% | 9.75 | true |
| D (fps 29.2) | 200, 381, 121, 313, 251, 57 | 23, 87, 11, 98, 30, 17 | 2.91% | 9.83 | true |

Independent 16-sample probe of the canvas crop (1247x801, 250 ms apart, `?drive=full`):
`warm` 2…274, `hot` 6…64, `max_r_minus_b` 167 to 171, **never 0**, with `spikeCount` 2 to 3 at every
sample (a separate 48-sample telemetry poll: `spikeCount` 2 to 3 continuously over 12 s).

So the accent reads in 12 of the last 13 cloud-live captures. **One earlier post-fix run read
`162, 14, 0, 0, 0, 0` and that is not explained**: `metrics.spikeCount = frame.spikes.length` is
the same array the flash consumes, so a sustained zero requires the drive to stop listing spikes,
which neither probe observed. Not reproduced in the three runs above; cause not pinned.
The accent's magnitude is phase-dependent (2 to 381 warm px), so it is legible, not constant.

Shader/colour changes that produced this: spike sprite size scaled by `aShock`
(`sizeScale = 1.0 + 2.0 * mag + 6.0 * aShock`, `minPx = 16.0` when `aShock > 0.02`), the spike
term gated as `clamp(vShock, 0.0, 1.0) * uGate`, the ring computed before the `alpha < 0.02`
coverage test so it carries its own alpha floor, and `uShockColor` `#f0a030` → `#ff9418`
(wider red-over-green margin in linear working space, where `#f0a030` sat ~0.51 against cyan's 0.49).

## Defect 2, dead-brain (`no_edges`) control

Controlled A/B on the single lever, dead frames only:

| `uQuiet` on a dead frame | no_edges lit | mean RGB | live/no_edges lit ratio |
|---|---|---|---|
| 0.055 | 12.77% |, | 2.23 |
| 0.012 (shipped) | 2.91% | (10, 15, 20) | 9.81 to 9.83 |

The 0.055 arm reproduces the previously recorded 12.9% / 2.19x almost exactly, so the earlier
reading predated the change and `uQuiet = 0.012` **is** applied in the panel
(`staticMat.uniforms.uQuiet.value = liveFrame ? 1 : 0.012`, every frame). 2.91% is inside the
2 to 3% target and comes from the ambient/anatomy term; live is untouched (`uQuiet` is 1.0 on a
live frame, `uAmbient` stays 0.1, `uGate` is live/dead only). The dead control's motion is
exactly 0 changed pixels between consecutive captures, versus ~19,000 (1.8 to 2.0%) live.

## Scenario flags (post-fix)

`nodes_visible` true · `amber_visible_live` true · `no_edges_dark` true · `trail_moving` true ·
`zero_props_renders` true · `live_vs_no_edges_lit_ratio` 9.83 · `live_vs_no_edges_mean_luma_ratio` 3.41.

`trail_moving` now reads true: `moved_pct` is 1.8 to 2.0% between consecutive live captures and
0 px in the dead control, so the trail genuinely moves and the earlier false was a threshold
artifact, not a dead trail. Points drawn 166,700; `caption_text` intact.
