/**
 * Mode badge for a live frame.
 *
 * The measured state magnitude is nearly identical under every live mode
 * (state_rms 0.105 intact, 0.108 shuffled, 0.096 random_graph), so the badge
 * is the only place the mode difference is visible. It carries the mode text
 * plus the accuracy the model reached under each control mode.
 */

import type { CSSProperties } from 'react';

const TONE: Record<string, { fg: string; bg: string; edge: string }> = {
  intact: { fg: '#7ff0c8', bg: 'rgba(46,160,126,0.16)', edge: 'rgba(127,240,200,0.45)' },
  shuffled: { fg: '#ffd479', bg: 'rgba(180,132,32,0.16)', edge: 'rgba(255,212,121,0.45)' },
  random_graph: { fg: '#c3a6ff', bg: 'rgba(120,88,220,0.16)', edge: 'rgba(195,166,255,0.45)' },
  no_edges: { fg: '#9fb2c4', bg: 'rgba(120,140,160,0.14)', edge: 'rgba(159,178,196,0.4)' },
  idle: { fg: '#8fa6bb', bg: 'rgba(90,110,130,0.14)', edge: 'rgba(143,166,187,0.35)' },
  unknown: { fg: '#9fb2c4', bg: 'rgba(120,140,160,0.14)', edge: 'rgba(159,178,196,0.4)' },
};

export interface ModeBadgeProps {
  mode: string;
  /** true when frames are coming from the socket, false for the idle path. */
  live: boolean;
  controls?: Record<string, number>;
}

export function ModeBadge({ mode, live, controls }: ModeBadgeProps) {
  const key = live ? mode : 'idle';
  const tone = TONE[key] ?? TONE.unknown;
  const chips: string[] = [];
  if (controls) {
    for (const name of ['shuffled_accuracy', 'no_edges_accuracy', 'random_graph_accuracy']) {
      const value = controls[name];
      if (typeof value !== 'number') continue;
      chips.push(`${name.replace('_accuracy', '')} ${(value * 100).toFixed(0)}%`);
    }
  }

  const style: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.4rem',
    padding: '0.18rem 0.5rem',
    borderRadius: '999px',
    border: `1px solid ${tone.edge}`,
    background: tone.bg,
    color: tone.fg,
    fontSize: '0.74rem',
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    whiteSpace: 'nowrap',
  };

  return (
    <span style={style} title={live ? 'mode reported by the live frame' : 'no live frames: synthetic idle animation'}>
      <span style={{ fontWeight: 700 }}>{live ? `mode ${mode}` : 'synthetic idle'}</span>
      {chips.length > 0 ? (
        <span style={{ opacity: 0.75, textTransform: 'none' }}>{chips.join(' / ')}</span>
      ) : null}
    </span>
  );
}

export default ModeBadge;
