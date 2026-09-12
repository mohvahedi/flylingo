#!/usr/bin/env bun
/**
 * Measure real frame rate for the viz-brain harness.
 *
 * It drives the installed Playwright chromium directly over the DevTools
 * protocol (no npm dependency), loads the built app, and samples the metrics
 * snapshot the app publishes on window.__bc. The number this prints is the
 * number the page showed while it was rendering, sampled while live frames
 * were arriving from ws://127.0.0.1:8770/stream.
 *
 * Usage:
 *   bun scripts/measure-fps.mjs                       # against http://127.0.0.1:4180/
 *   bun scripts/measure-fps.mjs --serve               # start vite preview itself
 *   bun scripts/measure-fps.mjs --modes subset,full --seconds 12
 *   bun scripts/measure-fps.mjs --uncapped            # disable vsync to see the raw cost
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/**
 * Kill a process and its children. Chrome and vite both re-exec themselves, so
 * killing only the launcher leaks a browser holding the devtools port and a
 * preview server holding 4180.
 */
function killTree(child) {
  if (!child || !child.pid) return;
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/T', '/F', '/PID', String(child.pid)], { stdio: 'ignore' });
  } else {
    try {
      process.kill(-child.pid, 'SIGKILL');
    } catch {
      try {
        child.kill('SIGKILL');
      } catch {
        // already gone
      }
    }
  }
}

// Whatever happens, do not leave a browser or a server behind.
const spawnedChildren = [];
process.on('exit', () => {
  for (const child of spawnedChildren) killTree(child);
});

const args = process.argv.slice(2);
function arg(name, fallback = null) {
  const i = args.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const next = args[i + 1];
  return next === undefined || next.startsWith('--') ? 'true' : next;
}
function flag(name) {
  return args.includes(`--${name}`);
}

const target = arg('url', 'http://127.0.0.1:4180/');
const seconds = Number(arg('seconds', '12'));
const warmup = Number(arg('warmup', '3'));
const modes = String(arg('modes', 'subset,full')).split(',').map((s) => s.trim()).filter(Boolean);
const uncapped = flag('uncapped');
const cdpPort = Number(arg('port', '9333'));
const sampleMs = 500;

function findChromium() {
  const explicit = process.env.CHROME_PATH;
  if (explicit && existsSync(explicit)) return explicit;

  const roots = [];
  const local = process.env.LOCALAPPDATA;
  if (local) roots.push(join(local, 'ms-playwright'));
  roots.push('C:/Program Files/Google/Chrome/Application');
  roots.push('C:/Program Files (x86)/Google/Chrome/Application');

  const found = [];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    const stack = [root];
    while (stack.length > 0) {
      const dir = stack.pop();
      let entries;
      try {
        entries = readdirSync(dir, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          if (entry.name === 'node_modules') continue;
          stack.push(full);
        } else if (entry.name === 'chrome.exe' || entry.name === 'headless_shell.exe') {
          found.push(full);
        }
      }
    }
  }
  // Prefer a full chrome build over the headless shell, then the shortest path.
  found.sort((a, b) => {
    const score = (p) => (p.includes('headless_shell') ? 1 : 0) * 1000 + p.length;
    return score(a) - score(b);
  });
  return found[0] ?? null;
}

class Cdp {
  constructor(url) {
    this.url = url;
    this.id = 0;
    this.pending = new Map();
    this.eventHandler = null;
    this.ws = new WebSocket(url);
    this.ready = new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve());
      this.ws.addEventListener('error', (event) => reject(new Error(`cdp socket error: ${event.message ?? 'unknown'}`)));
    });
    this.ws.addEventListener('message', (event) => {
      const msg = JSON.parse(typeof event.data === 'string' ? event.data : String(event.data));
      if (msg.id === undefined) {
        if (this.eventHandler) this.eventHandler(msg);
        return;
      }
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      this.pending.delete(msg.id);
      if (msg.error) entry.reject(new Error(`${entry.method}: ${msg.error.message}`));
      else entry.resolve(msg.result);
    });
  }

  send(method, params = {}, sessionId) {
    const id = (this.id += 1);
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.ws.send(JSON.stringify(payload));
    });
  }

  close() {
    try {
      this.ws.close();
    } catch {
      // already closed
    }
  }
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForCdp(port, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = 'no response';
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return await res.json();
      lastError = `HTTP ${res.status}`;
    } catch (error) {
      lastError = String(error?.message ?? error);
    }
    await sleep(250);
  }
  throw new Error(`devtools endpoint never came up on ${port}: ${lastError}`);
}

const EVAL_SNAPSHOT = `(() => {
  const m = window.__bc;
  if (!m) return JSON.stringify({ missing: true });
  return JSON.stringify({
    fps: m.fps, frames: m.frames, avgFrameMs: m.avgFrameMs, worstFrameMs: m.worstFrameMs,
    points: m.points, liveSlots: m.liveSlots, resolvedIds: m.resolvedIds,
    driveMode: m.driveMode, mappingBasis: m.mappingBasis, mode: m.mode, synthetic: m.synthetic,
    refValue: m.refValue, peakAbs: m.peakAbs, spikeCount: m.spikeCount,
    bytesPerFrame: m.bytesPerFrame, renderer: m.renderer,
  });
})()`;

const EVAL_TEXT = `(() => {
  const header = document.querySelector('header');
  const main = document.querySelector('main');
  const footer = document.querySelector('footer');
  const overlay = main ? main.innerText : '';
  return JSON.stringify({
    headerText: (header ? header.innerText : '').replace(/\\s*\\n\\s*/g, ' | '),
    overlayHead: overlay.split('\\n').slice(0, 6).join(' | '),
    anatomical: overlay.includes('measured soma voxels'),
    procedural: overlay.includes('procedural fallback'),
    strip: (footer ? footer.innerText : '').replace(/\\s*\\n\\s*/g, ' | '),
  });
})()`;

async function evaluate(cdp, sessionId, expression) {
  const result = await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: false }, sessionId);
  if (result.exceptionDetails) throw new Error(`evaluate failed: ${result.exceptionDetails.text}`);
  return result.result.value;
}

async function measure(cdp, url, label) {
  const { targetId } = await cdp.send('Target.createTarget', { url });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });

  // Collect the page's own errors and warnings while it renders.
  const events = [];
  cdp.eventHandler = (msg) => {
    if (msg.sessionId !== sessionId) return;
    if (msg.method === 'Runtime.consoleAPICalled') {
      const level = msg.params.type;
      if (level !== 'error' && level !== 'warning') return;
      const text = (msg.params.args ?? [])
        .map((a) => (a.value !== undefined ? String(a.value) : a.description ?? a.type))
        .join(' ');
      events.push({ level, text });
    } else if (msg.method === 'Log.entryAdded') {
      const entry = msg.params.entry;
      if (entry.level !== 'error' && entry.level !== 'warning') return;
      events.push({ level: entry.level, text: `${entry.source}: ${entry.text}${entry.url ? ` (${entry.url})` : ''}` });
    } else if (msg.method === 'Runtime.exceptionThrown') {
      events.push({ level: 'exception', text: msg.params.exceptionDetails?.text ?? 'exception' });
    }
  };
  await cdp.send('Runtime.enable', {}, sessionId);
  await cdp.send('Log.enable', {}, sessionId);

  const samples = [];
  const started = Date.now();
  let first = null;
  let last = null;

  while (Date.now() - started < (seconds + warmup) * 1000 + 2000) {
    const raw = await evaluate(cdp, sessionId, EVAL_SNAPSHOT);
    const m = raw ? JSON.parse(raw) : null;
    if (m && !m.missing) {
      if (!first) first = m;
      last = m;
      if (Date.now() - started >= warmup * 1000) samples.push(m);
      if (Date.now() - started >= warmup * 1000 + seconds * 1000) break;
    }
    await sleep(sampleMs);
  }

  const probeRaw = await evaluate(cdp, sessionId, EVAL_TEXT);
  const probe = probeRaw ? JSON.parse(probeRaw) : {};
  cdp.eventHandler = null;
  await cdp.send('Target.closeTarget', { targetId });

  const fps = samples.map((s) => s.fps).filter((v) => Number.isFinite(v) && v > 0);
  if (fps.length === 0) {
    return { label, error: 'no frames rendered: window.__bc.fps stayed 0', probe, first, last };
  }
  const mean = fps.reduce((a, b) => a + b, 0) / fps.length;
  const min = Math.min(...fps);
  const max = Math.max(...fps);
  const sorted = [...fps].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  return {
    label,
    fpsMean: mean,
    fpsMedian: median,
    fpsMin: min,
    fpsMax: max,
    samples: fps.length,
    seconds: seconds,
    frames: last ? last.frames - (first ? first.frames : 0) : 0,
    lastFps: last ? last.fps : 0,
    avgFrameMs: last ? last.avgFrameMs : 0,
    worstFrameMs: last ? last.worstFrameMs : 0,
    points: last ? last.points : 0,
    liveSlots: last ? last.liveSlots : 0,
    resolvedIds: last ? last.resolvedIds : 0,
    driveMode: last ? last.driveMode : 'unknown',
    mode: last ? last.mode : 'unknown',
    synthetic: last ? last.synthetic : null,
    refValue: last ? last.refValue : 0,
    peakAbs: last ? last.peakAbs : 0,
    spikeCount: last ? last.spikeCount : 0,
    bytesPerFrame: last ? last.bytesPerFrame : 0,
    renderer: last ? last.renderer : 'unknown',
    probe,
    events,
  };
}

async function main() {
  const chrome = findChromium();
  if (!chrome) {
    console.error('no chromium found: set CHROME_PATH or install playwright browsers');
    process.exit(2);
  }
  if (typeof WebSocket === 'undefined') {
    console.error('this runtime has no global WebSocket; run with bun');
    process.exit(2);
  }

  let preview = null;
  const previewUrl = target;
  if (flag('serve')) {
    // Start the preview server, but keep the caller's path and query so
    // ?stream=off and friends still select a scenario.
    preview = spawn('bun', ['run', 'preview'], { cwd: process.cwd(), stdio: 'ignore', shell: false });
    spawnedChildren.push(preview);
    const deadline = Date.now() + 30000;
    let up = false;
    while (Date.now() < deadline) {
      try {
        const res = await fetch(previewUrl);
        if (res.ok) {
          up = true;
          break;
        }
      } catch {
        // not listening yet
      }
      await sleep(300);
    }
    if (!up) {
      console.error(`preview server did not start on ${previewUrl}`);
      killTree(preview);
      process.exit(2);
    }
  }

  const profile = mkdtempSync(join(tmpdir(), 'vizbrain-fps-'));
  const flags = [
    '--headless=new',
    `--remote-debugging-port=${cdpPort}`,
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--window-size=1600,900',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    'about:blank',
  ];
  if (uncapped) flags.splice(1, 0, '--disable-frame-rate-limit', '--disable-gpu-vsync');

  const browser = spawn(chrome, flags, { stdio: 'ignore' });
  spawnedChildren.push(browser);
  const info = await waitForCdp(cdpPort);
  const cdp = new Cdp(info.webSocketDebuggerUrl);
  await cdp.ready;
  await cdp.send('Target.setDiscoverTargets', { discover: true });

  const results = [];
  for (const mode of modes) {
    const url = `${previewUrl}${previewUrl.includes('?') ? '&' : '?'}drive=${mode}`;
    const label = `${mode}${uncapped ? ' (uncapped)' : ' (vsync)'}`;
    process.stdout.write(`measuring ${label} ...\n`);
    results.push(await measure(cdp, url, label));
  }

  // Graceful browser shutdown first, then belt and braces on the launcher pid:
  // a hard kill of the launcher alone leaves the re-exec'd browser process
  // holding the devtools port.
  try {
    await cdp.send('Browser.close');
  } catch {
    // browser already gone
  }
  cdp.close();
  killTree(browser);
  killTree(preview);
  if (flag('json')) {
    console.log(JSON.stringify({ chrome, url: previewUrl, uncapped, results }));
    return;
  }

  for (const r of results) {
    console.log('');
    console.log(`=== ${r.label} ===`);
    if (r.error) {
      console.log(`  ERROR ${r.error}`);
      console.log(`  probe ${JSON.stringify(r.probe)}`);
      continue;
    }
    console.log(`  fps mean ${r.fpsMean.toFixed(2)} | median ${r.fpsMedian.toFixed(2)} | min ${r.fpsMin.toFixed(2)} | max ${r.fpsMax.toFixed(2)} over ${r.samples} samples / ${r.seconds}s after ${warmup}s warmup`);
    console.log(`  rendered frames in window ${r.frames} | avg frame ${r.avgFrameMs.toFixed(2)} ms | worst frame ${r.worstFrameMs.toFixed(2)} ms`);
    console.log(`  points ${r.points} | live slots ${r.liveSlots} | ids resolved ${r.resolvedIds} | drive ${r.driveMode} | bytes/frame ${r.bytesPerFrame}`);
    console.log(`  mode ${r.mode} | synthetic ${r.synthetic} | ref ${r.refValue.toFixed(4)} | peak |state| ${r.peakAbs.toFixed(4)} | spikes ${r.spikeCount}`);
    console.log(`  renderer ${r.renderer}`);
    console.log(`  console/log errors+warnings: ${(r.events ?? []).length}`);
    for (const e of (r.events ?? []).slice(0, 8)) console.log(`    [${e.level}] ${e.text.slice(0, 200)}`);
    console.log(`  dom  anatomical=${r.probe.anatomical} procedural=${r.probe.procedural}`);
    console.log(`  dom  overlay: ${r.probe.overlayHead}`);
    console.log(`  dom  footer: ${r.probe.strip}`);
  }
  console.log('');
  console.log(`chromium ${chrome}`);
  console.log(`url ${previewUrl}`);
}

main().catch((error) => {
  console.error(error?.stack ?? String(error));
  process.exit(1);
});
