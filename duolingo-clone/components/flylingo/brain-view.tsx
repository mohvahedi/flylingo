"use client";

import { useEffect, useState } from "react";

import dynamic from "next/dynamic";

import { ModeControls, TrainingBadge } from "@/components/flylingo/brain-panel";
import { useBrainStream, useServiceHealth } from "@/lib/flylingo/api";
import type { BrainFrame, Mode } from "@/lib/flylingo/types";

const BrainCloud = dynamic(
  () => import("@/components/viz/brain/components/BrainCloud"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center bg-neutral-950">
        <span className="animate-pulse text-xs font-bold uppercase tracking-wide text-neutral-500">
          loading brain cloud
        </span>
      </div>
    ),
  },
);

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-700 bg-neutral-900/80 px-2.5 py-1.5">
      <div className="text-[10px] font-bold uppercase tracking-wide text-neutral-500">
        {label}
      </div>
      <div className="font-mono text-sm text-neutral-200">{value}</div>
    </div>
  );
}

/**
 * Full-bleed connectome view.
 *
 * BrainCloud owns its own WebGL Canvas and positions its layers absolutely, so it needs
 * a real viewport rather than a panel slot. This route gives it one and supplies the
 * provenance text the component itself does not render.
 */
export function BrainView() {
  const { frame, status } = useBrainStream();
  const { health } = useServiceHealth();
  const [mea, setMea] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const measure = () =>
      setMea({ width: window.innerWidth, height: Math.max(360, window.innerHeight - 150) });
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const onModeChange = async (mode: Mode) => {
    try {
      await fetch(
        `${process.env.NEXT_PUBLIC_FLYLINGO_API ?? "http://127.0.0.1:8770"}/control`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        },
      );
    } catch {
      // The panel shows the mode the server reports, so a failed switch is visible.
    }
  };

  const f: BrainFrame | null = frame;

  return (
    <div className="flex min-h-screen flex-col bg-neutral-950 text-neutral-200">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-neutral-800 px-4 py-3">
        <h1 className="text-sm font-bold uppercase tracking-wide">
          MaleCNS v1.0, live
        </h1>
        <span
          className={
            "text-[10px] font-bold uppercase tracking-wide " +
            (status === "open" ? "text-green-500" : "text-amber-500")
          }
        >
          {status === "open" ? "stream open" : status}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <ModeControls
            mode={f?.mode ?? "intact"}
            onChange={(m) => void onModeChange(m)}
          />
        </div>
      </header>

      <div className="grid grid-cols-2 gap-2 px-4 py-3 sm:grid-cols-4 lg:grid-cols-8">
        <Stat label="Mode" value={f?.mode ?? "intact"} />
        <Stat label="Step" value={f ? `${f.step}` : "n/a"} />
        <Stat label="Active" value={f ? `${(f.active_fraction * 100).toFixed(2)}%` : "n/a"} />
        <Stat label="State rms" value={f ? f.state_rms.toFixed(4) : "n/a"} />
        <Stat label="Spikes" value={f ? `${f.spikes.length}` : "n/a"} />
        <Stat label="Neurons" value="166,700" />
        <Stat label="Edges" value="25,582,938" />
        <Stat label="Fly acc" value={f ? `${(f.fly_accuracy * 100).toFixed(0)}%` : "n/a"} />
      </div>

      {health && (
        <div className="px-4 pt-3">
          <TrainingBadge health={health} />
        </div>
      )}

      <div className="flex-1 px-4 pb-4">
        {mea ? (
          <div
            className="overflow-hidden rounded-xl border border-neutral-800"
            style={{ height: mea.height }}
          >
            <BrainCloud
              state={f?.state ?? []}
              spikes={f?.spikes ?? []}
              sampledIds={f?.sampled_ids ?? []}
              activeFraction={f?.active_fraction ?? 0}
              stateRms={f?.state_rms ?? 0}
              width={mea.width - 40}
              height={mea.height}
            />
          </div>
        ) : null}
      </div>

      <footer className="border-t border-neutral-800 px-4 py-3 text-[11px] leading-relaxed text-neutral-400">
        <div className="font-semibold text-neutral-300">
          Coordinates: measured soma voxels from the real connectome
        </div>
        <div>
          MaleCNS v1.0: 139,662 of the 166,700 retained neurons carry a measured soma
          annotation (<span className="font-mono">somaLocation</span>); 27,038 are placed
          at the centroid of their (class, in-degree decile) group because no soma was
          annotated. 25,582,938 directed edges; in-degree from the CSR row pointers
          (row = postsynaptic).
        </div>
        <div className="mt-1">
          Colour is auto-ranged to the 95th percentile of |state| in each frame, because the
          sampled activity is genuinely small (state rms about 0.105, p95 of |state| about
          0.274). Under the no-edges control every value is exactly zero and the view is
          genuinely empty rather than amplified.
        </div>
      </footer>
    </div>
  );
}
