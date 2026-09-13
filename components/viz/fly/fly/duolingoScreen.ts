/**
 * Draws the Duolingo lesson onto a canvas, for use as the texture on the phone's screen.
 *
 * Why a canvas rather than the real DOM components:
 *
 *  1. The fly has to WALK TO the answer it picked. That needs the pixel rectangle of every
 *     option card so the screen position can be converted to a world position on the phone.
 *     Rasterising the live DOM gives no such handle; drawing it here means the geometry is
 *     known exactly.
 *  2. It has to be a texture. The screen is a mesh in a 3D scene, lit and seen at an angle,
 *     so an HTML overlay would not sit on the phone convincingly.
 *  3. No rasteriser dependency and no font-inlining failure modes: the exact typeface is
 *     loaded with the FontFace API and every element is drawn deterministically.
 *
 * The design values are Duolingo's: the palette, the raised card (a 2px outline over a 4px
 * darker bottom edge), the 16px progress bar with its white top highlight, the numbered
 * badges, and the uppercase letter-spaced CTA. The typeface is Nunito, which is what the
 * duolingo-clone this project is built on uses, and the closest Google face to Duolingo's
 * own.
 */

export const SCREEN_W = 390;
export const SCREEN_H = 865;

/** Duolingo's palette. */
const C = {
  green: '#58CC02',
  greenDark: '#58A700',
  greenEdge: '#4CA302',
  blue: '#1CB0F6',
  blueDark: '#1899D6',
  bluePale: '#DDF4FF',
  bluePaleBorder: '#84D8FF',
  red: '#FF4B4B',
  redDark: '#EA2B2B',
  redPale: '#FFDFE0',
  redPaleBorder: '#FFC1C1',
  greenPale: '#D7FFB8',
  greenPaleBorder: '#A5ED6E',
  line: '#E5E5E5',
  /** Unselected cards get a darker bottom edge so the raised look actually reads. The
   *  earlier version used the same grey for outline and edge, which made every card flat. */
  lineEdge: '#D8D8D8',
  grey: '#AFAFAF',
  ink: '#4B4B4B',
  inkDark: '#3C3C3C',
  white: '#FFFFFF',
};

const FONT = 'Nunito, system-ui, sans-serif';
const PAD = 18;
/** the phone's own chrome at the top and the home indicator area at the bottom */
const STATUS_H = 30;
const SAFE_BOTTOM = 26;

export interface ScreenState {
  prompt: string;
  options: { text: string }[];
  /** index the FLY picked, -1 for none. Drawn as the live selection. */
  flyChoice: number;
  /** index the human picked, -1 for none */
  userChoice: number;
  status: 'none' | 'correct' | 'wrong';
  answerIndex: number;
  hearts: number;
  /** 0..1 */
  progress: number;
  /** lesson number within the unit, and the unit total, for the "Lesson 3 of 5" style subhead */
  lessonNo?: number;
  lessonTotal?: number;
  /** the clock shown in the phone's status bar */
  clock?: string;
}

/** Rectangles of the option cards, in screen pixels. The fly's walk targets come from here. */
export interface OptionRect {
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
}

let fontsPromise: Promise<void> | null = null;

/**
 * Load Nunito before the first draw. Canvas silently falls back to a default face if the
 * face is not ready, which would wreck the look, so every draw awaits this.
 */
export function ensureDuolingoFonts(): Promise<void> {
  if (fontsPromise) return fontsPromise;
  fontsPromise = (async () => {
    if (typeof document === 'undefined' || !('fonts' in document)) return;
    const load = async (weight: string, file: string) => {
      try {
        const face = new FontFace('Nunito', `url(/fonts/${file})`, { weight, style: 'normal' });
        await face.load();
        (document.fonts as FontFaceSet).add(face);
      } catch {
        // a missing weight degrades to the next one rather than breaking the screen
      }
    };
    await Promise.all([
      load('700', 'Nunito-700.ttf'),
      load('800', 'Nunito-800.ttf'),
      load('900', 'Nunito-900.ttf'),
    ]);
  })();
  return fontsPromise;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** A Duolingo card: fill, 2px outline, and a 4px bottom edge in a darker shade. */
function raisedCard(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  o: { fill: string; border: string; edge: string; radius?: number; edgeH?: number },
) {
  const r = o.radius ?? 12;
  const e = o.edgeH ?? 4;
  ctx.fillStyle = o.edge;
  roundRect(ctx, x, y + e, w, h, r);
  ctx.fill();
  ctx.fillStyle = o.fill;
  roundRect(ctx, x, y, w, h, r);
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = o.border;
  ctx.stroke();
}

function raisedButton(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  o: { fill: string; edge: string; text: string; radius?: number; fontSize?: number },
) {
  const r = o.radius ?? 14;
  const e = 4;
  ctx.fillStyle = o.edge;
  roundRect(ctx, x, y + e, w, h, r);
  ctx.fill();
  ctx.fillStyle = o.fill;
  roundRect(ctx, x, y, w, h, r);
  ctx.fill();
  ctx.fillStyle = o.text;
  ctx.font = `800 ${o.fontSize ?? 17}px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label.toUpperCase().split('').join('\u2009'), x + w / 2, y + h / 2 + 1);
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** A filled heart, drawn as a path so it stays crisp at any size. */
function heart(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(s / 24, s / 24);
  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.bezierCurveTo(0, 3, -2.5, 0, -6, 0);
  ctx.bezierCurveTo(-11, 0, -12, 5, -12, 7);
  ctx.bezierCurveTo(-12, 12, -7, 16, 0, 21);
  ctx.bezierCurveTo(7, 16, 12, 12, 12, 7);
  ctx.bezierCurveTo(12, 5, 11, 0, 6, 0);
  ctx.bezierCurveTo(2.5, 0, 0, 3, 0, 6);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/** The phone's own status bar, which is what makes the screen read as a phone. */
function statusBar(ctx: CanvasRenderingContext2D, clock: string) {
  ctx.fillStyle = C.white;
  ctx.fillRect(0, 0, SCREEN_W, STATUS_H);
  ctx.fillStyle = '#111111';
  ctx.font = `800 13px ${FONT}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(clock, 26, STATUS_H / 2 + 1);

  // signal bars, wifi wedge, battery: simple vector marks, the shapes the OS itself uses
  const x0 = SCREEN_W - 26;
  const cy = STATUS_H / 2;
  ctx.fillStyle = '#111111';
  for (let i = 0; i < 4; i += 1) {
    const h = 3.5 + i * 2.4;
    roundRect(ctx, x0 - 66 + i * 5.5, cy + 6 - h, 3.4, h, 1);
    ctx.fill();
  }
  // wifi
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#111111';
  ctx.beginPath();
  ctx.arc(x0 - 38, cy + 5, 8.5, Math.PI * 1.22, Math.PI * 1.78);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x0 - 38, cy + 5, 4.6, Math.PI * 1.22, Math.PI * 1.78);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x0 - 38, cy + 4.5, 1.3, 0, Math.PI * 2);
  ctx.fillStyle = '#111111';
  ctx.fill();
  // battery
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = 'rgba(17,17,17,0.5)';
  roundRect(ctx, x0 - 24, cy - 5, 22, 11, 3);
  ctx.stroke();
  ctx.fillStyle = '#111111';
  roundRect(ctx, x0 - 22, cy - 3, 15, 7, 1.6);
  ctx.fill();
  roundRect(ctx, x0 - 1, cy - 2, 1.8, 4, 0.8);
  ctx.fill();
}

/**
 * Draw the whole lesson screen. Returns the option rectangles so the scene can send the fly
 * to the one it picked.
 */
export function drawDuolingoLesson(
  ctx: CanvasRenderingContext2D,
  state: ScreenState,
): { options: OptionRect[] } {
  const W = SCREEN_W;
  const H = SCREEN_H;
  if (!ctx) return { options: [] };

  ctx.save();
  // Do NOT reset the transform: createScreenCanvas pre-scales the context so the screen can
  // be authored in 390x865 units and rendered at any device scale. An earlier version called
  // setTransform(1,0,0,1,0,0) here, which threw that scale away and drew the screen into the
  // top-left quarter of a 2x canvas with the rest transparent.
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = C.white;
  ctx.fillRect(0, 0, W, H);

  statusBar(ctx, state.clock ?? '9:41');

  // ---------------------------------------------------------------- lesson header
  const barY = STATUS_H + 22;

  ctx.strokeStyle = C.grey;
  ctx.lineWidth = 3.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(PAD, barY - 6);
  ctx.lineTo(PAD + 12, barY + 6);
  ctx.moveTo(PAD + 12, barY - 6);
  ctx.lineTo(PAD, barY + 6);
  ctx.stroke();

  const progX = PAD + 30;
  const progW = W - progX - PAD - 56;
  const progH = 16;
  const progY = barY - progH / 2;
  ctx.fillStyle = C.line;
  roundRect(ctx, progX, progY, progW, progH, progH / 2);
  ctx.fill();
  if (state.progress > 0) {
    const fillW = Math.max(progH, progW * Math.min(1, state.progress));
    ctx.fillStyle = C.green;
    roundRect(ctx, progX, progY, fillW, progH, progH / 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.34)';
    roundRect(ctx, progX + 5, progY + 3.5, Math.max(0, fillW - 10), 5, 3);
    ctx.fill();
  }

  ctx.fillStyle = C.red;
  heart(ctx, W - PAD - 26, barY - 1, 17);
  ctx.font = `800 17px ${FONT}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(state.hearts), W - PAD + 2, barY);

  // ---------------------------------------------------------------- prompt
  let y = STATUS_H + 108;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = C.inkDark;
  ctx.font = `800 29px ${FONT}`;
  for (const line of wrap(ctx, state.prompt, W - 44)) {
    ctx.fillText(line, W / 2, y);
    y += 37;
  }

  // ---------------------------------------------------------------- options
  // One column or two, decided from the actual text width rather than assumed. Duolingo
  // stacks these full width when the answers are phrases, and a half-width card truncates
  // them: at 20px "Buenas tardes" needs about 130px of the 171px a half card allows, and the
  // badge and padding take the rest. Measuring means a short answer set still gets the
  // compact two column grid and a long one gets the readable stack.
  const gap = 12;
  const half = (W - PAD * 2 - gap) / 2;
  const rowH = 76;
  ctx.font = `800 20px ${FONT}`;
  const widest = state.options.reduce((m, o) => Math.max(m, ctx.measureText(o.text).width), 0);
  const CHROME = 27 + 9 + 11 + 14; // badge, its gap, and the card's padding
  const cols = widest + CHROME <= half - 8 ? 2 : 1;
  const colW = cols === 1 ? W - PAD * 2 : half;
  const optTop = y + 22;
  const rects: OptionRect[] = [];

  state.options.forEach((opt, i) => {
    const col = cols === 1 ? 0 : i % 2;
    const row = cols === 1 ? i : Math.floor(i / 2);
    const x = PAD + col * (colW + gap);
    const oy = optTop + row * (rowH + gap);

    const isFlyPick = i === state.flyChoice;
    const isAnswer = i === state.answerIndex;
    const showResult = state.status !== 'none';

    let fill = C.white;
    let border = C.line;
    let edge = C.lineEdge;
    let textColour = C.ink;
    let badgeBorder = C.line;
    let badgeText = C.grey;

    if (showResult && (isAnswer || i === state.userChoice)) {
      if (isAnswer) {
        fill = C.greenPale;
        border = C.greenPaleBorder;
        edge = '#93DC59';
        textColour = '#3A7D0B';
        badgeBorder = C.greenPaleBorder;
        badgeText = '#3A7D0B';
      } else {
        fill = C.redPale;
        border = C.redPaleBorder;
        edge = '#FFB3B3';
        textColour = C.redDark;
        badgeBorder = C.redPaleBorder;
        badgeText = C.redDark;
      }
    } else if (isFlyPick) {
      // The fly's live pick, in Duolingo's own selected style: this is what a tapped option
      // looks like, which is exactly what the fly is doing.
      fill = C.bluePale;
      border = C.bluePaleBorder;
      edge = '#5BC8F5';
      textColour = C.blueDark;
      badgeBorder = C.bluePaleBorder;
      badgeText = C.blue;
    }

    raisedCard(ctx, x, oy, colW, rowH, { fill, border, edge });

    const bs = 27;
    const bx = x + 11;
    const by = oy + (rowH - bs) / 2;
    ctx.lineWidth = 2;
    ctx.strokeStyle = badgeBorder;
    roundRect(ctx, bx, by, bs, bs, 6);
    ctx.stroke();
    ctx.fillStyle = badgeText;
    ctx.font = `800 14px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), bx + bs / 2, by + bs / 2 + 0.5);

    ctx.fillStyle = textColour;
    ctx.font = `800 20px ${FONT}`;
    ctx.textAlign = 'left';
    const tx = bx + bs + 9;
    const maxW = colW - (tx - x) - 10;
    let label = opt.text;
    while (ctx.measureText(label).width > maxW && label.length > 3) label = label.slice(0, -2);
    if (label !== opt.text) label = `${label}\u2026`;
    ctx.fillText(label, tx, oy + rowH / 2 + 1);

    rects.push({ x, y: oy, w: colW, h: rowH, cx: x + colW / 2, cy: oy + rowH / 2 });
  });

  // ---------------------------------------------------------------- bottom action
  // Duolingo pins the CTA to the bottom of the screen; an earlier version let it float just
  // under the options, which left a dead area beneath it and is the most obvious tell that
  // the screen is not the real thing.
  const btnH = 58;
  const stripH = 104;

  if (state.status !== 'none') {
    // Duolingo's result footer: tinted band, message on the left, action on the right.
    const correct = state.status === 'correct';
    const stripY = H - stripH;
    ctx.fillStyle = correct ? C.greenPale : C.redPale;
    ctx.fillRect(0, stripY, W, stripH);
    ctx.fillStyle = correct ? C.greenPaleBorder : C.redPaleBorder;
    ctx.fillRect(0, stripY, W, 2);

    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = correct ? C.greenEdge : C.redDark;
    ctx.font = `900 21px ${FONT}`;
    ctx.fillText(correct ? 'Nicely done!' : 'Correct solution:', PAD, stripY + 44);
    ctx.font = `800 17px ${FONT}`;
    if (correct) {
      ctx.fillStyle = C.greenDark;
      ctx.fillText('+15 XP', PAD, stripY + 72);
    } else {
      ctx.fillStyle = C.redDark;
      const ans = state.options[state.answerIndex]?.text ?? '';
      let a = ans;
      while (ctx.measureText(a).width > W - PAD * 2 - 160 && a.length > 3) a = a.slice(0, -2);
      ctx.fillText(a, PAD, stripY + 72);
    }

    const bw = 150;
    const bx = W - PAD - bw;
    const by = stripY + (stripH - btnH) / 2 - 6;
    raisedButton(ctx, bx, by, bw, btnH, correct ? 'Continue' : 'Try again', {
      fill: correct ? C.green : C.red,
      edge: correct ? C.greenEdge : C.redDark,
      text: C.white,
    });
  } else {
    // The CTA reflects the FLY's pick, because the fly is the one playing: once it has chosen,
    // the button is live. Three design reviews flagged a disabled button next to a highlighted
    // answer as the single clearest sign the screen is fake, and they were right.
    const ready = state.flyChoice >= 0 || state.userChoice >= 0;
    const by = H - SAFE_BOTTOM - btnH;
    raisedButton(ctx, PAD, by, W - PAD * 2, btnH, 'Check', {
      fill: ready ? C.green : C.line,
      edge: ready ? C.greenEdge : C.line,
      text: ready ? C.white : C.grey,
    });
  }

  // home indicator
  ctx.fillStyle = 'rgba(0,0,0,0.30)';
  roundRect(ctx, W / 2 - 60, H - 11, 120, 5, 2.5);
  ctx.fill();

  ctx.restore();
  return { options: rects };
}

/** A canvas already sized and scaled, ready to hand to a CanvasTexture. */
export function createScreenCanvas(scale = 2): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = Math.round(SCREEN_W * scale);
  c.height = Math.round(SCREEN_H * scale);
  const ctx = c.getContext('2d');
  if (ctx) ctx.scale(scale, scale);
  return c;
}
