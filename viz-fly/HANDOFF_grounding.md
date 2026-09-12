# Handoff: grounding the fly specimen

**Status: UNSOLVED. The fly still reads as floating, not standing on a lit stage.**
Written 2026-09-13 so the machine could be shut down. The agent that was working on this was
stopped deliberately; it had not landed any source change.

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
