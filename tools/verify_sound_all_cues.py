"""Watch a long stretch of the lesson and tally every cue, to confirm the wrong-answer sound.

The fly runs at ~95% accuracy, so a short sample only ever shows correct.wav. This watches long
enough for misses to occur naturally rather than forcing one, which would mean either wiping the
trained brain or faking a result. It also confirms the finish fanfare fires on each new lesson.
"""
import sys
from collections import Counter

from playwright.sync_api import sync_playwright

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300

INSTRUMENT = r"""
window.__cues = [];
window.__spoken = [];
const realPlay = HTMLMediaElement.prototype.play;
HTMLMediaElement.prototype.play = function () {
  if (!this.muted) {
    const src = this.currentSrc || this.src || '';
    window.__cues.push(src.split('/').pop());
  }
  return realPlay.apply(this, arguments);
};
const realSpeak = window.speechSynthesis && speechSynthesis.speak.bind(speechSynthesis);
if (realSpeak) {
  speechSynthesis.speak = function (u) {
    window.__spoken.push(u.text);
    return realSpeak(u);
  };
}
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

    pg.evaluate(
        """() => {
          const b = [...document.querySelectorAll('button')]
            .find(x => /sound (on|off)/i.test(x.textContent || ''));
          if (b) b.click();
        }"""
    )
    print(f"sound enabled; watching {SECONDS}s of the lesson\n")

    seen = Counter()
    per_lesson = Counter()
    elapsed = 0
    while elapsed < SECONDS:
        pg.wait_for_timeout(10000)
        elapsed += 10
        r = pg.evaluate(
            """() => ({
              cues: window.__cues.slice(),
              spoken: window.__spoken.slice(),
              lesson: (() => {
                const t = document.body.innerText || '';
                const m = /lesson (\\d+)\\s*\\/\\s*(\\d+)/i.exec(t);
                return m ? m[1] + '/' + m[2] : null;
              })(),
            })"""
        )
        for c in r["cues"]:
            seen[c] += 1
        r["cues"].clear()
        pg.evaluate("() => { window.__cues = []; }")
        if r["lesson"]:
            per_lesson[r["lesson"]] += 1
        print(
            f"  t={elapsed:>3}s  correct={seen['correct.wav']:>3}  wrong={seen['incorrect.wav']:>3}  "
            f"finish={seen['finish.mp3']:>2}  phrases={sum(seen.values()) - seen['finish.mp3']:>3}  "
            f"lesson {r['lesson']}"
        )

    final = pg.evaluate("() => ({ spoken: window.__spoken.slice() })")
    br.close()

print()
print("cue tally:")
for k in ("correct.wav", "incorrect.wav", "finish.mp3"):
    print(f"  {k:16s} {seen[k]}")
print()
print(f"  spanish phrases spoken: {len(final['spoken'])}")
uniq = sorted(set(final["spoken"]))
print(f"  distinct phrases      : {len(uniq)}")
for t in uniq[:14]:
    print(f"     {t!r}")
print()
print(f"  lessons visited       : {sorted(per_lesson)}")
print(f"  console errors        : {errs[:3] if errs else 'none'}")
print()
ok = seen["correct.wav"] > 0 and seen["incorrect.wav"] > 0 and seen["finish.mp3"] > 0
print("RESULT:", "all three Duolingo cues fired" if ok else "a cue never fired")
sys.exit(0 if ok else 1)
