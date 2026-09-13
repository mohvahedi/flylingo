"""Does the demo run itself, and does the course actually advance?

This is the claim the user cares about: the fly plays on its own, answers, earns reward and
moves through the lessons. It is checked by watching the live page for a stretch and reading the
panels out of the DOM, not by inspecting code.
"""
import sys
import time

from playwright.sync_api import sync_playwright

DURATION_S = 90

READ = """() => {
  const grab = (re) => {
    for (const e of document.querySelectorAll('*')) {
      const t = (e.textContent || '').trim();
      if (e.children.length === 0 && re.test(t)) return t;
    }
    return null;
  };
  const all = [...document.querySelectorAll('*')].map(e => (e.textContent || '').trim());
  const find = (re) => all.find(t => re.test(t)) || null;
  return {
    answered: find(/^\\d+ \\/ \\d+ answered$/i),
    accuracy: find(/^\\d+(\\.\\d+)?%$/),
    lesson: find(/^lesson \\d+ \\/ \\d+$/i),
    lessonTitle: find(/^(Greetings|Travel|Family|Food|Numbers|Colors|Animals|Weather|Time|Shopping|Office|Body|Clothes|Verbs|Phrases)$/),
    learnt: find(/^\\d+\\/\\d+$/),
    rehearsals: grab(/^[\\d,]+$/),
    dopaminePulses: find(/^\\d+ pulses$/i),
    flyAcc: find(/^\\d+%$/),
    step: (() => {
      const lab = [...document.querySelectorAll('*')].find(e =>
        (e.textContent||'').trim()==='step' && !e.children.length);
      return lab && lab.parentElement ? lab.parentElement.textContent.trim() : null;
    })(),
    headline: (() => {
      const h = document.querySelector('h2');
      return h ? h.textContent.trim() : null;
    })(),
  };
}"""

with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    )
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:150]))
    pg.on("console", lambda m: errs.append("console: " + m.text[:120]) if m.type == "error" else None)

    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)
    pg.wait_for_timeout(20000)  # boot + first answer cycle

    seen = []
    t0 = time.time()
    while time.time() - t0 < DURATION_S:
        seen.append(pg.evaluate(READ))
        pg.wait_for_timeout(6000)

    print(f"sampled {len(seen)} times over {DURATION_S}s\n")
    print(f"{'t':>4} {'answered':>12} {'lesson':>14} {'learnt':>9} {'pulses':>8} {'step':>6}")
    print("-" * 62)
    for i, s in enumerate(seen):
        t = i * 6
        print(
            f"{t:>4} {str(s['answered']):>12} {str(s['lesson']):>14} "
            f"{str(s['learnt']):>9} {str(s['dopaminePulses']):>8} {str(s['step']):>6}"
        )

    first, last = seen[0], seen[-1]
    print()
    ans = lambda s: int(str(s["answered"]).split("/")[0]) if s["answered"] and "/" in str(s["answered"]) else 0
    lesson_n = lambda s: int(str(s["lesson"]).split()[1]) if s["lesson"] else 0

    progressed = ans(last) > ans(first)
    advanced_lesson = lesson_n(last) > lesson_n(first)
    got_pulses = (last["dopaminePulses"] or "0") != (first["dopaminePulses"] or "0")

    print(f"answered advanced      : {ans(first)} -> {ans(last)}   {'YES' if progressed else 'NO'}")
    print(f"lesson advanced        : {lesson_n(first)} -> {lesson_n(last)}   {'YES' if advanced_lesson else 'NO'}")
    print(f"dopamine pulses moved  : {first['dopaminePulses']} -> {last['dopaminePulses']}   {'YES' if got_pulses else 'NO'}")
    print(f"headline seen          : {first['headline']!r} -> {last['headline']!r}")
    print(f"lesson title           : {last['lessonTitle']!r}")
    print(f"console errors         : {errs[:4] if errs else 'none'}")
    print()
    ok = progressed or advanced_lesson
    print("RESULT:", "the demo runs itself and advances" if ok else "STILL NOT PROGRESSING")
    br.close()
    sys.exit(0 if ok else 1)
