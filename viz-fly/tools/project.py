"""Print the live camera transform and project known world points to canvas pixels, so the
measurement boxes in the other probes can be placed from geometry instead of guesswork.

    python tools/project.py [url]
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5191/?frozen'
LAUNCH_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']

JS = r"""
() => {
  const s = window.__flyScene;
  if (!s) return { error: 'no scene' };
  let cam = null;
  s.traverse((o) => { if (o.isPerspectiveCamera && o.fov && o.parent === s) cam = o; });
  if (!cam) s.traverse((o) => { if (o.isPerspectiveCamera && o.fov && !/cube/i.test(o.type)) cam = o; });
  let camList = [];
  s.traverse((o) => { if (o.isCamera) camList.push([o.type, o.name, o.position.toArray().map(n=>+n.toFixed(2)), o.fov || null, o.parent === s, !!o.isPerspectiveCamera]); });
  if (!cam) return { error: 'no camera', camList };
  const v = new (cam.position.constructor)();
  const box = new (window.__flyScene.constructor === Object ? Object : Object)();
  const proj = (x, y, z) => {
    v.set(x, y, z).project(cam);
    return [Math.round((v.x * 0.5 + 0.5) * 1440), Math.round((-v.y * 0.5 + 0.5) * 900)];
  };
  const out = {
    cam: cam ? cam.position.toArray().map(n => +n.toFixed(3)) : null,
    camList: camList,
    camQ: cam ? cam.quaternion.toArray().map(n => +n.toFixed(3)) : null,
    fov: cam ? cam.fov : null,
    target: (() => { const t = new (cam.position.constructor)(); cam.getWorldDirection(t);
      return t.toArray().map(n => +n.toFixed(3)); })(),
    pts: {},
    objs: [],
  };
  for (const [x, y, z] of [[0,0,0],[1,0,0],[-1,0,0],[0,0,1],[0,0,-1],[0,0,3],[0,0,6.9],
                           [3.4,0,0],[0,0.5,0],[0,0.9,0],[0,-0.4,0],[0,0.5,-1.5],[1.5,0,1.5]]) {
    out.pts[x + ',' + y + ',' + z] = proj(x, y, z);
  }
  const bb = new (Object.getPrototypeOf(cam.position).constructor === Object ? Object : Object)();
  const B = cam.position.constructor;
  for (const name of ['fly-fit','ground','ring-inner','ring-outer']) {
    const o = s.getObjectByName(name);
    if (!o) { out.objs.push([name, 'missing']); continue; }
    o.updateWorldMatrix(true, true);
    const b = new (window.__THREEBox || Object)();
    out.objs.push([name, o.visible,
      o.position.toArray().map(n => +n.toFixed(3)),
      (() => { try { o.geometry.computeBoundingBox();
        const g = o.geometry.boundingBox;
        return [g.min.toArray().map(n=>+n.toFixed(2)), g.max.toArray().map(n=>+n.toFixed(2))]; }
        catch (e) { return String(e).slice(0, 60); } })()]);
  }
  return out;
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH_ARGS)
    page = b.new_page(viewport={'width': 1440, 'height': 900})
    page.goto(URL, wait_until='load', timeout=45000)
    page.wait_for_selector('canvas', timeout=45000)
    page.wait_for_timeout(4500)
    page.evaluate("""() => { const c = document.querySelector('canvas');
        if (c && c.parentElement) for (const el of Array.from(c.parentElement.children))
          if (el.tagName !== 'CANVAS') el.style.display = 'none'; return 1; }""")
    res = page.evaluate(JS)
    for k, v in res.items():
        print(k, '=', v)
    b.close()
