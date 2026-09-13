# Handoff: grounding the fly specimen

**Status: PARTLY SOLVED — the fly reads as grounded; the bounded edge is now measurable but was**
**not visible until the rim fix below.** See "Round 2" at the end of this file for what changed
after this header was written. The original header, kept because the diagnosis below still stands:

> Status as of the first pass: UNSOLVED. The fly still reads as floating, not standing on a lit stage.
> Written 2026-09-13 so the machine could be shut down. The agent that was working on this was
> stopped deliberately; it had not landed any source change.

## The problem

A vision reviewer repeatedly reports "the fly floats in a void". The fly.glb specimen loads and
animates correctly, 1,911 setae are present, bloom and tonemap are correct, and the shadow rig
is geometrically correct. The failure is purely that the shadow does not *read*.

## What is measured right now (reproducible)

Run `bun run dev` in `viz-fly` (port 5191), then:

```
cd D:/Projects/flylingo/viz-fly
D:/Projects/flylingo/brain/.venv/Scripts/python.exe tools/ground_measure.py
```

`?noshadow` renders the identical frame with the whole shadow rig switched off, so the delta
isolates shadowing. Latest numbers:

| Measure | Value |
|---|---|
| luma under the body, shadows ON | **20.38** |
| luma under the body, shadows OFF | **47.91** |
| shadow effect under the body | **57.5% darker** |
| luma of the slab around the shadow | **24.54** |
| **local contrast: shadow vs surrounding slab** | **17.0% darker** |

**This is the number that matters, and it is the reason the fly still reads as floating.** The
absolute darkening (57.5%) is strong, but the shadow is only 17% darker than the slab
immediately next to it, and the eye judges grounding by *local* contrast, not absolute
darkening. The reference image achieves a clearly readable pooled shadow. Raise this number
substantially, not the absolute one.

`void` boxes measured `nan` — the sample boxes fall outside the frame. Fix the box coordinates
in `ground_measure.py` before trusting that row.

## What has already been tried (do not simply revert)

- Ground colour raised `#141c25` -> `#3a4657` in `src/fly/materials.ts` (`groundMaterial`).
  Rationale: a surface too dark to see cannot show a shadow. This took the under-body darkening
  from 11.6% to 51%.
- Footprint fade relaxed `smoothstep(7.2, 2.0, r)` -> `smoothstep(7.8, 3.4, r)` so the pool stays
  lit further out and the far rim still falls to black.
- Slate, not gloss: `roughness` 0.4 -> 0.58, `clearcoat` 0.45 -> 0.22, to read as a matte desk
  slab rather than wet plastic.
- Per-tarsus contact patch alpha 0.62 -> 0.80 and thorax pool 0.40 -> 0.55 in `src/fly/contacts.ts`.
- drei `ContactShadows` opacity 0.5 -> 0.75, `far` 1.0, `scale` 3.4.
- Tone mapping is ACES filmic, exposure 0.95; bloom strength 0.24, radius 0.4, threshold 1.3.

## The actual target

`C:/Users/Lion/AppData/Roaming/Hermes/composer-images/image_de7a8a.png` is the user's reference.
It is a **lit dark-studio render**: the fly stands on a *distinct, clearly lit slate slab* that
reads as a physical object with a gradient toward its edges, with a soft-edged contact shadow
pooled under the legs, key light from the **upper-left**, and a near-black void only *behind* the
slab. A dark frame containing a lit object — not an infinitely faded dark plane.

## The next step a successor should take

The strongest hypothesis, not yet tried: **stop making the ground an infinite faded disc.** Make
it a distinct slab or stage under the fly with its own visible edge and gradient, lit by the key,
with the void behind it. Tune until `ground_measure.py` reports the local contrast row well above
its current 17%, and confirm visually, not only numerically — the numbers and the reviewer have
disagreed before, so check both.

## Instruments that now exist (reusable)

- `viz-fly/tools/ground_measure.py` — the shadows-on/shadow-off A/B, per-box luma, local contrast,
  and a vertical luma profile. The authoritative measurement.
- `viz-fly/tools/finemap.py` — fine-grained ground map (untested; was mid-run when stopped).
- `viz-fly/tools/verify_model.py` — the broader model harness (animation states, setae, glow,
  fallback, fps). Note its contact-shadow block was fixed to bound by both `feet` and
  `translations`, and to read z at index 1 of a 2-element `[x, z]` row.
- `artifacts/viz-fly-ground-on.png`, `viz-fly-ground-off.png`, `viz-fly-hero.png` — current state.

## Must stay intact

- The CC-BY-4.0 attribution for the model author **victorberdugo1** (in the HUD, `ATTRIBUTION.txt`,
  and README.md). The model is a third-party asset and the licence requires the credit.
- The rig that drives the model's 57 named nodes: six-leg walk gait and the selective two-leg
  groom pass, both measured.
- `setae` = 1,911 bristles.
- Bloom/tonemap: blown-out pixels 1.89% -> 0.0001%, p99 luma 245 -> 174.
- Do not touch `brain/`, `duolingo-clone/`, or `viz-brain/`.

## Environment notes

- Use `D:/Projects/flylingo/brain/.venv/Scripts/python.exe` for anything needing Pillow or
  Playwright; it has both. It has **no `pip` module** — install with
  `cd D:/Projects/flylingo/brain && VIRTUAL_ENV=.venv uv pip install <pkg>`.
- The `brain` venv's Playwright browsers are already present under `LOCALAPPDATA/ms-playwright`.
- Never use `readPixels` for evidence: it returns zeros without `preserveDrawingBuffer`. Use
  element or full-page screenshots analysed as PNG bytes, which is what all tools here do.
- Do not rebuild while `next start` is running — see README.md.

## Round 2 — what actually fixed it (2026-09-13)

The ground is now a bounded extruded slab (`src/fly/slab.ts`) with a lit rim band, and the fly
reads as grounded: an independent reviewer looking at the live render said the shadow "along with
the leg placement, makes the fly read as grounded rather than floating" — the first time that has
been true. The two real causes were not the ones the earlier pass chased. **First**, drei's
`ContactShadows` was not casting a contact shadow at all: measured with the fly hidden so nothing
could cast, the floor under it read 12.2 luma against 44.8 with `?noshadow`, and switching the key
light's shadow map off changed nothing (12.20 -> 12.19). It was painting its whole 3.4 x 3.4 plane
near black, and that uniform black quad *was* the "void" the fly floated over. It was deleted; the
cast shadow is now three's own `ShadowMaterial` overlay (`ground-shadow` in `FlyStage.tsx`), whose
alpha is `opacity * (1 - getShadowMask())`, so the depth of the shadow is art directed instead of
being diluted by the fill, the bounce and the environment. **Second**, the old `fade * fade` = 0 at
r=3.4 on geometry that ran to r=7 meant the surface was already black well before its own edge, so
the boundary was the tail of a gradient rather than an edge.

**The bounded edge: fixed, and here is the honest measurement.** A reviewer had said the stage
"fades into black rather than ending at a clear edge", and that was correct. The rim now brightens
instead of darkening — a band from 0.80 to 1.0 of the half extents that climbs back to ~1.25x the
slab's centre value, so the boundary is a lit edge against a 0-luma void. `tools/slab_edge.py`
reads the first lit pixel down each column: the slab/void step is now **median 86.7 luma, min 48.7,
max 171** across fly-free columns, where the old faded rim sat at ~9 luma just inside the edge.
Measured with `tools/ground_measure.py` (three instrument bugs fixed, see below):

| Measure | Before | Now |
|---|---|---|
| local contrast, pool under body vs slab beside it | 17.0% (wrong box — see below) | **55.0% darker** |
| shadow rig, under-body on/off | 57.5% | 12.2% darker (28.76 / 32.76) |
| void, both upper corners | `nan` | **0.00 / 0.00** |
| slab lit value beside the fly | 24.54 | 63.91 |

`tools/ground_local.py` locates the darkening from the data instead of a box pair and ablates the
pool live through `uPoolStrength`: pool **21.3% local contrast**, shadow rig **43.1%** (52.5% darker
where it lands), both 1.7x and 3.2x the temporal noise floor. Zero console errors in every run.

**One thing I could not do, so it is not claimed.** The framing advice to "bring the near edge into
frame" is not achievable at this camera, and the skirt is not the lever either. Ray casting the
frame against the slab's box shows the frame's bottom edge crosses the floor at world
`z = 1.087 - 0.786x`, so the floor is only visible from ~2.1 units in front of the camera outward —
a near edge can only be shown by pulling `hz` below ~1.1, inside the fly's own footprint (feet reach
z=+0.9, pool reaches z=+1.5). And **no wall of the box is visible at any thickness**: the far wall
is hidden under the top face, the left wall faces away from a camera at x=+1.85, and the right and
near walls are below the frame. Thickness was ray cast at 0.16, 0.20 and 0.34 with no change. The
skirt is still held at full albedo for any other framing, but from the hero camera the edge can only
be carried by the top face's silhouette, which is why the rim band is the fix. Slab half extents are
now 2.30 x 2.10 (was 3.35 x 2.95) so the left edge lands at screen x~98 with black beyond it rather
than at x=-76 with one pixel of margin, and the scale rings were scaled to 0.96 / 1.52 to match.

**Three instrument bugs fixed, because each had produced a confident wrong number.** In
`ground_measure.py`: the `BOX` table was documented `(x0,x1,y0,y1)` and unpacked `(y0,y1,x0,x1)`, so
the "under" and "slab" boxes both sampled the near floor *beside* the legs — **the 17.0% local
contrast in the table above was that measurement, not a shadow**; `HIDE_JS` hid the canvas's
siblings, which is the canvas itself, so the HUD stayed in every shot; and the void boxes fell
outside the frame, which is why that row read `nan`. In `ground_local.py`: the A/B footprint had the
wrong sign, and the fly's own bright pixels were allowed into the footprint, which is why an early
ablation appeared to show *turning shadows off* making the scene brighter. The fly silhouette is now
derived by differencing a fly-hidden frame and excluded from every footprint.

