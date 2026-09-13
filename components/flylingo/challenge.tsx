"use client";

import { cn } from "@/lib/utils";

import { Card } from "./card";

import type { CardType } from "@/lib/flylingo/types";

type ChallengeProps = {
  options: string[];
  onSelect: (index: number) => void;
  status: "correct" | "wrong" | "none";
  selectedOption?: number;
  disabled?: boolean;
  type: CardType;
  /** Spoken Spanish for each option, when the challenge has audio. */
  audioSrc?: (string | null)[];
};

/** The clone's option grid, keyed by option index rather than a database id. */
export const Challenge = ({
  options,
  onSelect,
  status,
  selectedOption,
  disabled,
  type,
  audioSrc,
}: ChallengeProps) => {
  return (
    <div
      className={cn(
        "grid gap-2",
        type === "ASSIST" && "grid-cols-1",
        type === "SELECT" && "grid-cols-2 lg:grid-cols-[repeat(auto-fit,minmax(0,1fr))]"
      )}
    >
      {options.map((option, i) => (
        <Card
          key={`${i}-${option}`}
          id={i}
          text={option}
          imageSrc={null}
          audioSrc={audioSrc?.[i] ?? null}
          shortcut={`${i + 1}`}
          selected={selectedOption === i}
          onClick={() => onSelect(i)}
          status={status}
          disabled={disabled}
          type={type}
        />
      ))}
    </div>
  );
};
