/**
 * Standalone viewer for the phone scene, so the composition can be judged and the flight
 * verified without the app around it.
 *
 * Served by Vite at /phone.html. The controls below exist to drive the exact states the
 * demo will hit: which card the fly picks, and the correct / wrong result views. The
 * synthetic activity matches the measured real magnitudes (p95 of |activity| near 0.27) so
 * the glow behaves as it does in the app.
 */
import { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';

import { PhoneStage } from './fly/PhoneStage';
import { STATE_LEN, syntheticFrame } from './fly/regions';
import type { OptionRect } from './fly/duolingoScreen';

const QUESTIONS: { prompt: string; options: string[]; answer: number }[] = [
  {
    prompt: "How do you say 'Hello' in Spanish?",
    options: ['Buenas tardes', 'Hola', 'Buenas noches', 'Buenos días'],
    answer: 1,
  },
  {
    prompt: "How do you say 'Thank you' in Spanish?",
    options: ['Por favor', 'Gracias', 'De nada', 'Lo siento'],
    answer: 1,
  },
  {
    prompt: "How do you say 'Good morning' in Spanish?",
    options: ['Buenos días', 'Buenas noches', 'Hasta luego', 'Buenas tardes'],
    answer: 0,
  },
  {
    prompt: "How do you say 'The apple' in Spanish?",
    options: ['El pan', 'La manzana', 'El agua', 'La leche'],
    answer: 1,
  },
];

/**
 * Presentation mode, set by ?embed=1 when this page is shown inside another page.
 *
 * The controls exist to drive the scene at exact states while judging the composition, which
 * is not what a viewer of the embed wants to see. With them hidden the scene drives itself.
 */
const EMBED = new URLSearchParams(window.location.search).get('embed') === '1';

function Viewer() {
  const [qi, setQi] = useState(0);
  const [flyChoice, setFlyChoice] = useState(1);
  const [status, setStatus] = useState<'none' | 'correct' | 'wrong'>('none');
  const [userChoice, setUserChoice] = useState(-1);
  const [hearts, setHearts] = useState(5);
  const [t, setT] = useState(0);
  const [rects, setRects] = useState<OptionRect[]>([]);

  // a synthetic activity frame so the fly's glow and motion are live offline
  const [activity, setActivity] = useState<number[]>(new Array(STATE_LEN).fill(0));
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      if (dt > 0.02) {
        setT((v) => v + dt);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    const f = syntheticFrame(t);
    setActivity(f.state);
  }, [t]);

  /**
   * Embed mode, for showing this scene on a page with nobody to click the controls.
   *
   * The fly picks the correct answer, the result resolves, then the next question loads. The
   * beats match the recorded clip served beside it: 3s to land and register, 6.2s per question.
   */
  useEffect(() => {
    if (!EMBED) return;
    const q = QUESTIONS[qi];
    setFlyChoice(q.answer);
    setUserChoice(-1);
    setStatus('none');
    const resolve = window.setTimeout(() => {
      setStatus('correct');
      setUserChoice(q.answer);
    }, 3000);
    const advance = window.setTimeout(() => setQi((v) => (v + 1) % QUESTIONS.length), 6200);
    return () => {
      window.clearTimeout(resolve);
      window.clearTimeout(advance);
    };
  }, [qi]);

  const q = QUESTIONS[qi];
  // memoised: an inline .map() would hand the scene a new array every render
  const options = useMemo(() => q.options.map((text) => ({ text })), [q]);
  const progress = (qi + (status === 'none' ? 0 : 1)) / QUESTIONS.length;

  const btn = (label: string, on: boolean, onClick: () => void) => (
    <button
      key={label}
      type="button"
      onClick={onClick}
      style={{
        padding: '5px 10px',
        marginRight: 6,
        background: on ? '#5BC8D6' : 'transparent',
        color: on ? '#080C11' : '#9AA8B6',
        border: `1px solid ${on ? '#5BC8D6' : 'rgba(120,160,200,0.3)'}`,
        borderRadius: 3,
        font: 'inherit',
        fontSize: 11,
        cursor: 'pointer',
      }}
    >
      {label}
    </button>
  );

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#080c11' }}>
      <PhoneStage
        prompt={q.prompt}
        options={options}
        flyChoice={flyChoice}
        userChoice={userChoice}
        status={status}
        answerIndex={q.answer}
        hearts={hearts}
        progress={progress}
        activity={activity}
        stateRms={0.11}
        activeFraction={0.02}
        onScreen={({ rects: r }) => setRects(r)}
      />
      {!EMBED && (
      <div
        style={{
          position: 'absolute',
          left: 12,
          bottom: 12,
          padding: '10px 12px',
          background: 'rgba(8,12,17,0.86)',
          border: '1px solid rgba(120,160,200,0.28)',
          borderRadius: 4,
          fontSize: 11,
          lineHeight: 1.9,
          maxWidth: 560,
        }}
      >
        <div>
          <span style={{ color: '#74828F', letterSpacing: '0.14em' }}>FLY PICKS </span>
          {[0, 1, 2, 3].map((i) => btn(`${i + 1}`, flyChoice === i, () => setFlyChoice(i)))}
        </div>
        <div>
          <span style={{ color: '#74828F', letterSpacing: '0.14em' }}>STATUS </span>
          {btn('open', status === 'none', () => { setStatus('none'); setUserChoice(-1); })}
          {btn('correct', status === 'correct', () => {
            setStatus('correct');
            setUserChoice(flyChoice);
          })}
          {btn('wrong', status === 'wrong', () => {
            setStatus('wrong');
            setUserChoice((flyChoice + 1) % 4);
            setHearts((h) => Math.max(0, h - 1));
          })}
        </div>
        <div>
          <span style={{ color: '#74828F', letterSpacing: '0.14em' }}>QUESTION </span>
          {btn('prev', false, () => setQi((v) => (v + QUESTIONS.length - 1) % QUESTIONS.length))}
          {btn('next', false, () => setQi((v) => (v + 1) % QUESTIONS.length))}
          <span style={{ color: '#5BC8D6' }}> {qi + 1}/{QUESTIONS.length}</span>
        </div>
        <div style={{ color: '#74828F', marginTop: 4 }}>
          rects: {rects.length} · card {flyChoice + 1} centre{' '}
          {rects[flyChoice] ? `${Math.round(rects[flyChoice].cx)},${Math.round(rects[flyChoice].cy)}` : ', '}
        </div>
      </div>
      )}
    </div>
  );
}

const el = document.getElementById('root');
if (el) createRoot(el).render(<Viewer />);