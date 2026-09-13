/**
 * Dev-only inspection strip. It renders ONLY when the host passes a frame (activity
 * present) and only when explicitly enabled, so the production surface stays clean.
 *
 * Purpose: let a human force a single animation and watch it in isolation. Nothing here
 * changes what the app sends; it only overrides which behavior the rig plays.
 */
import type { Behavior } from '../fly/pose';

export type ControlStripProps = {
  behavior: Behavior;
  source: 'auto' | 'manual';
  override: Behavior | null;
  onOverride: (b: Behavior | null) => void;
  paused: boolean;
  onTogglePause: () => void;
  timeScale: number;
  onTimeScale: (v: number) => void;
  onTrigger: (kind: 'celebrate' | 'recoil') => void;
};

const BEHAVIORS: Array<{ id: Behavior; label: string }> = [
  { id: 'idle', label: 'idle' },
  { id: 'walk', label: 'walk' },
  { id: 'groom', label: 'groom' },
  { id: 'proboscis', label: 'proboscis' },
  { id: 'startle', label: 'startle' },
];

const btn = (active: boolean): React.CSSProperties => ({
  background: active ? 'rgba(34,211,238,0.18)' : 'rgba(15,23,42,0.72)',
  border: `1px solid ${active ? 'rgba(34,211,238,0.7)' : 'rgba(148,163,184,0.28)'}`,
  color: active ? '#a5f3fc' : '#cbd5e1',
  font: 'inherit',
  fontSize: 11,
  padding: '3px 8px',
  borderRadius: 4,
  cursor: 'pointer',
  letterSpacing: '0.02em',
});

export function ControlStrip({
  behavior,
  source,
  override,
  onOverride,
  paused,
  onTogglePause,
  timeScale,
  onTimeScale,
  onTrigger,
}: ControlStripProps) {
  return (
    <div
      style={{
        position: 'absolute',
        left: 12,
        bottom: 12,
        display: 'flex',
        gap: 6,
        alignItems: 'center',
        flexWrap: 'wrap',
        maxWidth: 'calc(100% - 24px)',
        background: 'rgba(2,6,23,0.66)',
        border: '1px solid rgba(148,163,184,0.24)',
        borderRadius: 6,
        padding: '6px 8px',
        backdropFilter: 'blur(6px)',
      }}
    >
      <span style={{ fontSize: 10, color: '#64748b', marginRight: 2 }}>dev</span>
      {BEHAVIORS.map((b) => (
        <button
          key={b.id}
          type="button"
          style={btn(override === b.id)}
          onClick={() => onOverride(override === b.id ? null : b.id)}
        >
          {b.label}
        </button>
      ))}
      <button type="button" style={btn(override === null)} onClick={() => onOverride(null)}>
        auto
      </button>
      <span style={{ width: 8 }} />
      <button type="button" style={btn(paused)} onClick={onTogglePause}>
        {paused ? 'resume' : 'freeze'}
      </button>
      <label style={{ fontSize: 10, color: '#64748b', display: 'flex', alignItems: 'center', gap: 4 }}>
        speed
        <input
          type="range"
          min={0.1}
          max={2}
          step={0.05}
          value={timeScale}
          onChange={(e) => onTimeScale(Number(e.target.value))}
          style={{ width: 68, accentColor: '#22d3ee' }}
        />
        <span style={{ color: '#94a3b8', minWidth: 26 }}>{timeScale.toFixed(2)}x</span>
      </label>
      <span style={{ width: 8 }} />
      <button type="button" style={btn(false)} onClick={() => onTrigger('celebrate')}>
        celebrate
      </button>
      <button type="button" style={btn(false)} onClick={() => onTrigger('recoil')}>
        recoil
      </button>
      <span style={{ fontSize: 10, color: '#475569', marginLeft: 4 }}>
        playing {behavior} ({source})
      </span>
    </div>
  );
}
