"""Diagnose why the Duolingo screen is not rendering in Nunito.

The canvas silently falls back to a default face if the FontFace is not ready, so this
checks each stage separately rather than assuming.
"""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    br = p.chromium.launch(headless=True)
    pg = br.new_page()
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text[:200]}"))
    pg.on("requestfailed", lambda r: logs.append(f"FAILED {r.url} {r.failure}"))
    pg.goto("http://localhost:5191/", wait_until="load", timeout=40000)

    out = pg.evaluate(
        """async () => {
          const res = {};
          // 1. can the browser even fetch the font file?
          try {
            const r = await fetch('/fonts/Nunito-800.ttf');
            res.fontFetch = r.status + ' ' + r.headers.get('content-type');
            const buf = await r.arrayBuffer();
            res.fontBytes = buf.byteLength;
          } catch (e) { res.fontFetch = 'ERR ' + String(e).slice(0,120); }

          // 2. does FontFace load, and what does it say if not?
          try {
            const f = new FontFace('Nunito', 'url(/fonts/Nunito-800.ttf)', {weight:'800'});
            const loaded = await f.load();
            res.fontFaceLoaded = true;
            res.fontFaceStatus = loaded.status;
            document.fonts.add(loaded);
          } catch (e) {
            res.fontFaceLoaded = false;
            res.fontFaceError = String(e && (e.message||e)).slice(0,200);
          }

          // 3. does the face actually win when measured?
          await document.fonts.ready;
          res.checkNunito = document.fonts.check('800 17px Nunito');
          res.facesInDoc = [...document.fonts].map(f => f.family + ':' + f.weight + ':' + f.status);

          // 4. measure the same string in Nunito vs a known different font
          const c = document.createElement('canvas');
          const ctx = c.getContext('2d');
          const t = 'Buenas noches';
          ctx.font = '800 17px Nunito, sans-serif';
          const wNunito = ctx.measureText(t).width;
          ctx.font = '800 17px monospace';
          const wMono = ctx.measureText(t).width;
          ctx.font = '800 17px "NoSuchFontXYZ", sans-serif';
          const wFallback = ctx.measureText(t).width;
          res.widthNunito = Math.round(wNunito*100)/100;
          res.widthMono = Math.round(wMono*100)/100;
          res.widthFallback = Math.round(wFallback*100)/100;
          res.nunitoDiffersFromFallback = Math.abs(wNunito - wFallback) > 0.5;
          return res;
        }"""
    )
    for k, v in out.items():
        print(f"  {k}: {v}")
    print()
    print("console / network:")
    for l in logs[:15]:
        print("   ", l)
    br.close()
