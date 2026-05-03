"use client";

import { useEffect, useRef, useState } from "react";

export interface UseTypewriterOptions {
  enabled: boolean;
  charsPerSecond?: number;
  maxLag?: number;
}

export function useTypewriter(
  target: string,
  { enabled, charsPerSecond = 80, maxLag = 300 }: UseTypewriterOptions,
): string {
  const [displayed, setDisplayed] = useState(0);
  const targetRef = useRef(target);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number | null>(null);

  useEffect(() => {
    targetRef.current = target;
  });

  useEffect(() => {
    if (!enabled) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      lastTickRef.current = null;
      return;
    }

    const tick = (now: number) => {
      if (lastTickRef.current === null) lastTickRef.current = now;
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;

      setDisplayed((prev) => {
        const targetLen = targetRef.current.length;
        if (targetLen < prev) return 0;
        const lag = targetLen - prev;
        if (lag <= 0) return prev;
        const speedMultiplier = lag > maxLag ? 1 + (lag - maxLag) / maxLag : 1;
        const advance = Math.max(1, Math.round(charsPerSecond * speedMultiplier * dt));
        return Math.min(targetLen, prev + advance);
      });

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      lastTickRef.current = null;
    };
  }, [enabled, charsPerSecond, maxLag]);

  return enabled ? target.slice(0, displayed) : target;
}
