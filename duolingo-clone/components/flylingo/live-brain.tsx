"use client";

import dynamic from "next/dynamic";
import Link from "next/link";

import type { BrainFrame } from "@/lib/flylingo/types";

/**
 * The 3D fly, loaded client-side only.
 *
 * It uses WebGL through react-three-fiber, which has no server renderer. Loading it
 * dynamically keeps it out of the server bundle. The brain cloud was moved to its own
 * full-page route (/lesson/fly/brain) because BrainCloud owns its own Canvas and needs
 * a real viewport, rather than a panel slot where its layers get clipped.
 */
const FlyStage = dynamic(() => import("@/components/viz/fly/FlyStage"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-neutral-900">
      <span className="animate-pulse text-[10px] font-bold uppercase tracking-wide text-neutral-500">
        loading fly
      </span>
    </div>
  ),
});

export function LiveBrain({
  frame,
  correct,
  height = 220,
}: {
  frame: BrainFrame | null;
  correct: boolean | null;
  height?: number;
}) {
  const state = frame?.state ?? [];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-bold uppercase tracking-wide text-neutral-400">
          Fly
        </span>
        <span className="text-[10px] text-neutral-400">
          {frame ? "live" : "synthetic idle"}
        </span>
        <Link
          href="/lesson/fly/brain"
          className="ml-auto text-[10px] font-bold uppercase tracking-wide text-sky-600 hover:underline"
        >
          Brain cloud
        </Link>
        <Link
          href="/fly"
          className="text-[10px] font-bold uppercase tracking-wide text-sky-600 hover:underline"
        >
          Broadcast view
        </Link>
      </div>

      <div className="overflow-hidden rounded-xl bg-neutral-900" style={{ height }}>
        <FlyStage
          activity={state}
          stateRms={frame?.state_rms ?? 0}
          activeFraction={frame?.active_fraction ?? 0}
          correct={correct}
          reward={frame?.reward ?? 0}
          mode={frame?.mode ?? "intact"}
          height={height}
        />
      </div>
    </div>
  );
}
