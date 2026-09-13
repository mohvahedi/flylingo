import type { Metadata } from "next";

import { BrainView } from "@/components/flylingo/brain-view";

export const metadata: Metadata = {
  title: "FlyLingo: connectome view",
  description:
    "All 166,700 neurons of the MaleCNS fruit fly connectome, live, with per-neuron activity and spikes.",
};

export default function BrainPage() {
  return <BrainView />;
}
