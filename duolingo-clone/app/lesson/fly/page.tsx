import type { Metadata } from "next";

import { FlyLingoQuiz } from "@/components/flylingo/quiz";

export const metadata: Metadata = {
  title: "FlyLingo: teach a fruit fly Spanish",
  description:
    "A real MaleCNS fruit-fly connectome choosing answers in a Duolingo-style Spanish lesson, with live neuron activity.",
};

/**
 * The FlyLingo lesson.
 *
 * The shell is the duolingo-clone's lesson layout (header, prompt, option grid,
 * footer); the data comes from the local brain service rather than Clerk/Drizzle,
 * because the lesson state here is the connectome's activity, not a user record.
 */
export default function FlyLessonPage() {
  return (
    <div className="flex h-full flex-col">
      <FlyLingoQuiz />
    </div>
  );
}
