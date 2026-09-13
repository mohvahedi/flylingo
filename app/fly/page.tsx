import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { HudApp } from "@/components/flylingo/hud/app";

/**
 * The unified HUD: fly body, fly connectome, and the Duolingo lesson in one frame.
 *
 * Composed as a 16:9 broadcast stage. Two typefaces, matching the reference: a neutral
 * grotesque for the UI and a monospace for anything that reads as data or a command.
 */
const inter = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "FlyLingo: broadcast view",
  description:
    "A real MaleCNS fruit-fly connectome answering a Duolingo-style Spanish lesson, with the specimen and the connectome live in one frame.",
};

export default function FlyBroadcastPage() {
  return (
    <div
      className={`${inter.variable} ${mono.variable}`}
      style={{
        fontFamily: "var(--font-sans), system-ui, -apple-system, Segoe UI, sans-serif",
        background: "#080C11",
      }}
    >
      <HudApp />
    </div>
  );
}
