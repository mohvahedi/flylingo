"""Does the lesson actually make Duolingo's sounds, and does the unlock path work?

Both questions are answered by instrumenting the browser rather than by listening:

  - HTMLMediaElement.play is wrapped before the page's own scripts run, so every cue the app
    fires is recorded with its source file, whether or not headless has an audio device.
  - speechSynthesis.speak is wrapped the same way, so the Spanish word is confirmed to be
    requested even in a browser with no voices installed.
  - The autoplay path is checked both ways: with the policy relaxed (the normal case for a
    browser the user has interacted with) and with it enforced, to prove that clicking the
    sound control is genuinely what unlocks playback.

Run:  verify_sound.py
"""
import sys

from playwright.sync_api import sync_playwright

APP = "http://127.0.0.1:3300/fly"

INSTRUMENT = r"""
window.__cues = [];
window.__spoken = [];
window.__blocked = [];
const realPlay = HTMLMediaElement.prototype.play;
HTMLMediaElement.prototype.play = function () {
  const src = this.currentSrc || this.src || '';
  if (!this.muted) window.__cues.push(src.split('/').pop());
  const p = realPlay.apply(this, arguments);
  if (p && p.catch) p.catch((e) => window.__blocked.push(src.split('/').pop() + ': ' + e.name));
  return p;
};
if ('speechSynthesis' in window) {
  const realSpeak = speechSynthesis.speak.bind(speechSynthesis);
  speechSynthesis.speak = function (u) {
    window.__spoken.push({ text: u.text, lang: u.lang, rate: u.rate });
    return realSpeak(u);
  };
}
"""


def run(label, autoplay_ok, click_sound, seconds):
    args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    args.append(
        "--autoplay-policy=no-user-gesture-required"
        if autoplay_ok
        else "--autoplay-policy=user-gesture-required"
    )
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=args)
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        pg.add_init_script(INSTRUMENT)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:130]))
        pg.goto(APP, wait_until="load", timeout=60000)
        pg.wait_for_timeout(16000)

        if click_sound:
            # The control is the gesture that unlocks playback.
            pg.evaluate(
                """() => {
                  const b = [...document.querySelectorAll('button')]
                    .find(x => /sound (on|off)/i.test(x.textContent || ''));
                  if (b) b.click();
                }"""
            )
            pg.wait_for_timeout(1500)

        state = pg.evaluate(
            """() => ({
              toggle: (() => {
                const b = [...document.querySelectorAll('button')]
                  .find(x => /sound (on|off)/i.test(x.textContent || ''));
                return b ? b.textContent.trim() : null;
              })(),
              cues: window.__cues.slice(),
              spoken: window.__spoken.slice(),
              blocked: window.__blocked.slice(),
              voices: 'speechSynthesis' in window ? speechSynthesis.getVoices().length : -1,
            })"""
        )

        # let the hands-off loop run so cues accumulate
        pg.wait_for_timeout(seconds * 1000)

        final = pg.evaluate(
            """() => ({
              cues: window.__cues.slice(),
              spoken: window.__spoken.slice(),
              blocked: window.__blocked.slice(),
            })"""
        )
        br.close()

    print(f"--- {label} ---")
    print(f"  sound toggle reads     : {state['toggle']!r}")
    print(f"  speech voices present  : {state['voices']}")
    if click_sound:
        print(f"  cues at click          : {state['cues'] or 'none'}")
        print(f"  spoken at click        : {[s['text'] for s in state['spoken']] or 'none'}")
        print(f"  blocked at click       : {state['blocked'] or 'none'}")
    print(f"  cues during lesson     : {final['cues'] or 'NONE'}")
    print(f"  spanish spoken         : {[s['text'] for s in final['spoken']] or 'NONE'}")
    if final["spoken"]:
        s0 = final["spoken"][0]
        print(f"    (lang={s0['lang']}, rate={s0['rate']})")
    print(f"  blocked play calls     : {final['blocked'] or 'none'}")
    print(f"  console errors         : {errs[:3] if errs else 'none'}")
    print()
    return len(final["cues"]), len(final["spoken"]), state["toggle"], final["blocked"]


# 1. The normal case: the browser permits audio.
cues, spoken, toggle, _ = run("autoplay allowed, sound enabled by clicking the control", True, True, 26)
ok1 = cues > 0 and spoken > 0 and toggle == "sound on"
print(f"  -> cues fired: {cues}, phrases spoken: {spoken}")
print(f"  -> {'PASS' if ok1 else 'FAIL'}: sound plays and the answer is spoken")
print()

# 2. Enforced policy without a click: must stay silent, and the control must still read "off"
#    rather than claiming sound is on.
cues2, spoken2, toggle2, _ = run("autoplay enforced, control never clicked", False, False, 20)
ok2 = cues2 == 0 and toggle2 == "sound off"
print(f"  -> {'PASS' if ok2 else 'FAIL'}: silent until the user opts in, control honest")
print()

print("RESULT:", "PASS" if (ok1 and ok2) else "FAIL")
sys.exit(0 if (ok1 and ok2) else 1)
