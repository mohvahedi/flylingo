"""Sound on, long run: does the app keep going, and does the wrong-answer cue fire?

Two questions, and they need separating:

 1. The app's progress counter is sampled alongside the cue tally, so "cues stopped" can be told
    apart from "the app stopped". A previous run showed no new cues after ~100s, which is
    ambiguous without this.

 2. The wrong-answer cue is provoked the honest way: the control switches the fly off, a HUMAN
    picks a deliberately wrong option, and CHECK submits it. That drives the same wrong-answer
    code path the fly uses on a miss, without wiping the trained brain to force one.
"""
import sys
from collections import Counter

from playwright.sync_api import sync_playwright

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 150

INSTRUMENT = r"""
window.__cues = [];
const realPlay = HTMLMediaElement.prototype.play;
HTMLMediaElement.prototype.play = function () {
  if (!this.muted) {
    const src = this.currentSrc || this.src || '';
    window.__cues.push(src.split('/').pop());
  }
  return realPlay.apply(this, arguments);
};
"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=[
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    pg.add_init_script(INSTRUMENT)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:130]))
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(14000)

    seen = Counter()
    pg.evaluate(
        """() => {
          const b = [...document.querySelectorAll('button')]
            .find(x => /sound (on|off)/i.test(x.textContent || ''));
          if (b) b.click();
        }"""
    )
    pg.wait_for_timeout(1200)
    print(f"sound enabled. phase 1: hands-off for {SECONDS}s\n")
    print(f"{'t':>5} {'answered':>10} {'fly acc':>8}  {'correct':>7} {'wrong':>5} {'finish':>6}  app alive")
    print("-" * 68)

    elapsed = 0
    prev_answered = None
    while elapsed < SECONDS:
        pg.wait_for_timeout(10000)
        elapsed += 10
        r = pg.evaluate(
            """() => {
              const cues = window.__cues.slice();
              window.__cues = [];
              const txt = document.body.innerText || '';
              const a = /(\\d+)\\s*\\/\\s*\\d+\\s*answered/i.exec(txt);
              const acc = [...document.querySelectorAll('*')]
                .map(e => (e.textContent||'').trim())
                .find(t => /^\\d+%$/.test(t));
              return { cues, answered: a ? Number(a[1]) : null, acc };
            }"""
        )
        for c in r["cues"]:
            seen[c] += 1
        alive = "" if prev_answered is None else ("YES" if r["answered"] != prev_answered else "no change")
        prev_answered = r["answered"]
        print(
            f"{elapsed:>5} {str(r['answered']):>10} {str(r['acc']):>8}  "
            f"{seen['correct.wav']:>7} {seen['incorrect.wav']:>5} {seen['finish.mp3']:>6}  {alive}"
        )

    # ---- phase 2: provoke a wrong answer through the human path
    print("\nphase 2: pausing the fly and answering wrongly on purpose\n")
    pg.evaluate(
        """() => {
          const b = [...document.querySelectorAll('button')]
            .find(x => /fly is playing|paused/i.test(x.textContent || ''));
          if (b) b.click();
        }"""
    )
    pg.wait_for_timeout(1500)

    # Pausing leaves the panel in whatever state the fly's last answer created. If that state is
    # "correct", CHECK is acting as NEXT, so clicking it only advances and submits nothing --
    # which is what made an earlier version of this test report no wrong-answer cue. Clear the
    # state first by pressing the button until it reads "check" again.
    for _ in range(4):
        label = pg.evaluate(
            """() => {
              const b = [...document.querySelectorAll('button')]
                .find(x => /^(check|next|retry)$/i.test((x.textContent||'').trim()));
              return b ? b.textContent.trim().toLowerCase() : null;
            }"""
        )
        if label == "check":
            break
        pg.evaluate(
            """() => {
              const b = [...document.querySelectorAll('button')]
                .find(x => /^(check|next|retry)$/i.test((x.textContent||'').trim()));
              if (b && !b.disabled) b.click();
            }"""
        )
        pg.wait_for_timeout(1200)
    print(f"  action button now reads: {label!r}")

    # select an option that is NOT the fly's pick (the fly's own probability is the percentage
    # shown, so anything but the highest is a different choice), then submit it separately.
    picked = pg.evaluate(
        """() => {
          const tiles = [...document.querySelectorAll('button')]
            .filter(b => /\\d+\\.\\d+%/.test(b.textContent || ''));
          if (!tiles.length) return null;
          const pcts = tiles.map(t => parseFloat((/(\\d+\\.\\d+)%/.exec(t.textContent||'')||[0,0])[1]));
          const best = pcts.indexOf(Math.max(...pcts));
          const idx = pcts.findIndex((_, i) => i !== best);
          if (idx < 0) return null;
          tiles[idx].click();
          return (tiles[idx].textContent || '').trim().slice(0, 24);
        }"""
    )
    print(f"  selected a wrong option: {picked!r}")
    pg.wait_for_timeout(1200)

    submitted = pg.evaluate(
        """() => {
          const check = [...document.querySelectorAll('button')]
            .find(b => /^(check|next|retry)$/i.test((b.textContent||'').trim()));
          if (!check) return 'no check button';
          if (check.disabled) return 'check was disabled';
          if (check.textContent.trim().toLowerCase() !== 'check') return 'button read ' + check.textContent.trim();
          check.click();
          return 'submitted';
        }"""
    )
    print(f"  check button: {submitted}")
    pg.wait_for_timeout(4000)

    # Which way did it go? Picking a "not the fly's choice" option does NOT guarantee a wrong
    # answer -- on a fill-the-blank the fly's lowest-probability option is often the right word,
    # which is why an earlier attempt at this looked like a missing cue. So read the outcome the
    # UI reports and check the cue that should accompany it.
    outcome = pg.evaluate(
        """() => {
          const t = document.body.innerText || '';
          if (/not yet/i.test(t)) return 'wrong';
          if (/\\bcorrect\\b/i.test(t)) return 'correct';
          return null;
        }"""
    )
    print(f"  outcome the UI reported: {outcome!r}")

    after = pg.evaluate("() => window.__cues.slice()")
    for c in after:
        seen[c] += 1
    br.close()

print()
print("cue tally for the whole run:")
for k in ("correct.wav", "incorrect.wav", "finish.mp3"):
    print(f"  {k:16s} {seen[k]}")
print()
print(f"console errors: {errs[:3] if errs else 'none'}")
ok = seen["correct.wav"] > 0 and seen["incorrect.wav"] > 0
print()
print("RESULT:", "correct AND wrong cues both fire" if ok else "a cue never fired")
sys.exit(0 if ok else 1)
