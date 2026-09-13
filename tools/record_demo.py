"""Record a 45-second video of the FlyLingo demo actually playing.

Video only: there is no system-audio loopback device on this machine and Playwright's recorder
captures no audio, so the file is silent. The sound control is deliberately left OFF in the
recording rather than shown as "on", so the video does not imply an audio track it does not have.

Recording starts the moment the browser context is created, which includes the ~15s the app
needs to boot the connectome and reach the first question. That dead head is trimmed off with
ffmpeg, using the wall-clock offset at which the lesson genuinely started, so the delivered clip
begins on live gameplay rather than on a loading screen.

Output: artifacts/flylingo-45s.webm (raw) and artifacts/flylingo-45s.mp4 (H.264 + faststart,
the form X accepts).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"D:\Projects\flylingo\artifacts")
#: Output stem, overridable so a recording can be taken while a previous clip is still open in a
#: player. Windows locks a file that is being played, and overwriting it in place fails with
#: WinError 32; that is not a reason to force-close whatever the user has open.
STEM = sys.argv[1] if len(sys.argv) > 1 else "flylingo-45s"
RAW = OUT / f"{STEM}.webm"
MP4 = OUT / f"{STEM}.mp4"
CLIP_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 45
W, H = 1920, 1080

OUT.mkdir(parents=True, exist_ok=True)
for f in (RAW, MP4):
    if f.exists():
        try:
            f.unlink()
        except PermissionError:
            print(f"{f.name} is open in another process; writing alongside it instead")
            ts = time.strftime("%H%M%S")
            RAW = OUT / f"{STEM}-{ts}.webm"
            MP4 = OUT / f"{STEM}-{ts}.mp4"
            break

t0 = time.time()
with sync_playwright() as p:
    br = p.chromium.launch(
        headless=True,
        args=[
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            # keep the tab rendering at full speed while offscreen
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
        ],
    )
    ctx = br.new_context(
        viewport={"width": W, "height": H},
        record_video_dir=str(OUT / "_rec"),
        record_video_size={"width": W, "height": H},
        device_scale_factor=1,
    )
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:130]))
    pg.goto("http://127.0.0.1:3300/fly", wait_until="load", timeout=60000)

    # Wait until the lesson is genuinely live: the connectome has booted and the fly has begun
    # answering. This is the point the clip should start from.
    live_at = None
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            answered = pg.evaluate(
                """() => {
                  const m = /(\\d+)\\s*\\/\\s*\\d+\\s*answered/i.exec(document.body.innerText || '');
                  return m ? Number(m[1]) : 0;
                }"""
            )
        except Exception:
            answered = 0
        if answered and answered > 0:
            live_at = time.time() - t0
            break
        pg.wait_for_timeout(500)

    if live_at is None:
        print("lesson never became live; aborting")
        ctx.close()
        br.close()
        sys.exit(1)

    print(f"lesson live at t={live_at:.1f}s -> clip starts here")

    # Sample the stage's own probe while recording, so the delivered clip can be checked against
    # ground truth instead of against a viewer's impression of a 470px contact sheet. PhoneStage
    # publishes window.__flyPos every frame; this records the reach extension, how many tarsi are
    # on the chosen card, and which card is being flown to, at the same wall-clock times the video
    # is being written. That is what makes it possible to point at a second in the clip and say
    # whether the fly was planted on its answer.
    READ = """() => {
      const p = window.__flyPos;
      if (!p) return null;
      const legs = p.legs || [];
      return { t: p.t, behavior: p.behavior, moving: p.moving, dist: p.dist,
               planted: p.reach ? p.reach.planted : 0, on: p.reach ? p.reach.on : 0,
               onCard: legs.filter((l) => l.onCard).length, target: p.target };
    }"""
    probe = []

    # record from live_at for CLIP_SECONDS, plus a small tail
    target = live_at + CLIP_SECONDS + 1.5
    while time.time() - t0 < target:
        s = pg.evaluate(READ)
        if s:
            s["clip_t"] = round(time.time() - t0 - live_at, 2)  # seconds into the delivered clip
            probe.append(s)
        pg.wait_for_timeout(200)

    # capture the state at the end so the clip can be described truthfully
    final = pg.evaluate(
        """() => {
          const txt = document.body.innerText || '';
          const a = /(\\d+)\\s*\\/\\s*(\\d+)\\s*answered/i.exec(txt);
          const l = /lesson (\\d+)\\s*\\/\\s*(\\d+)/i.exec(txt);
          const acc = [...document.querySelectorAll('*')].map(e => (e.textContent||'').trim()).find(t => /^\\d+%$/.test(t));
          return { answered: a ? a[0] : null, lesson: l ? l[0] : null, flyAcc: acc };
        }"""
    )
    video_path = Path(pg.video.path())
    ctx.close()
    br.close()

raw_duration = time.time() - t0
print(f"recorded {raw_duration:.1f}s of browser session")
print(f"final state: {final}")
print(f"console errors: {errs[:3] if errs else 'none'}")

# write the probe alongside the video, so the clip can be verified rather than described
PROBE = OUT / f"{STEM}-probe.json"
PROBE.write_text(
    json.dumps({"clip_seconds": CLIP_SECONDS, "live_at": live_at,
                "samples": probe}, indent=1),
    encoding="utf-8",
)
planted = [s for s in probe if s["planted"] >= 0.9]
on_card = [s for s in probe if s["planted"] >= 0.9 and s["onCard"] >= 1]
print(f"probe: {len(probe)} samples, {len(planted)} with the forelegs planted, "
      f"{len(on_card)} of those on the chosen card")
print(f"wrote {PROBE.name}")

# move the recorder's file out of its temp dir, then trim the boot head
RAW.write_bytes(video_path.read_bytes())
for f in (OUT / "_rec").glob("*"):
    f.unlink()
(OUT / "_rec").rmdir()

print(f"raw video: {RAW.name} ({RAW.stat().st_size/1024:.0f} KB)")

cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-ss", f"{live_at:.3f}",
    "-i", str(RAW),
    "-t", str(CLIP_SECONDS),
    "-c:v", "libx264", "-preset", "slow", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-an",
    str(MP4),
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("ffmpeg failed:", r.stderr[-600:])
    sys.exit(1)

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
     "-show_entries", "stream=width,height,codec_name,nb_frames",
     "-of", "default=nw=1", str(MP4)],
    capture_output=True, text=True,
)
print()
print("delivered:", MP4)
print(probe.stdout.strip())
print(f"clip starts at {live_at:.1f}s of the session, runs {CLIP_SECONDS}s, silent by design")
