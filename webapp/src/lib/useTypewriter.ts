"use client";

import { useEffect, useRef, useState } from "react";

export interface UseTypewriterOptions {
  enabled: boolean;
  charsPerSecond?: number;
  maxLag?: number;
}

/**
 * Paces the visible reveal of streamed text. While `enabled` is true, advances
 * `displayed` toward `target.length` at ~charsPerSecond, with mild speed-up
 * when the lag exceeds `maxLag` so we don't fall arbitrarily far behind.
 *
 * Drain-to-target: when `enabled` flips from true→false (stream just ended),
 * the typewriter keeps ticking at the same rate until it has caught up to
 * `target.length`, then stops. Without this, the end-of-stream snap turns
 * "almost-streamed" into a visible pop — which is exactly what users saw
 * on Fast mode when the backend buffered the response and dumped it all
 * at once just before close-frame.
 *
 * Mount-time gate: when the hook mounts with `enabled=false` (historical
 * messages loaded from the DB on chat-open), we initialize `displayed` to
 * `target.length` so the bubble appears in full immediately — no animation.
 * Only newly-streaming bubbles (mount with `enabled=true`) start at zero
 * and reveal character-by-character. This matches the OpenAI/Claude UI:
 * old turns pop in, only the live one types.
 */
export function useTypewriter(
  target: string,
  { enabled, charsPerSecond = 80, maxLag = 300 }: UseTypewriterOptions,
): string {
  // Mount-time gate: historical bubbles (enabled=false at mount) snap to
  // full immediately; live bubbles (enabled=true at mount) start at 0 and
  // animate. The lazy initializer reads `enabled`/`target` once at mount.
  const [displayed, setDisplayed] = useState<number>(() =>
    enabled ? 0 : target.length,
  );
  const targetRef = useRef(target);
  const enabledRef = useRef(enabled);
  const displayedRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number | null>(null);

  // Keep refs current so the rAF closure always reads fresh values without
  // resubscribing the effect every render.
  useEffect(() => { targetRef.current = target; });
  useEffect(() => { enabledRef.current = enabled; });
  useEffect(() => { displayedRef.current = displayed; });

  useEffect(() => {
    const tick = (now: number) => {
      if (lastTickRef.current === null) lastTickRef.current = now;
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;

      const targetLen = targetRef.current.length;
      let caughtUp = false;

      setDisplayed((prev) => {
        if (targetLen < prev) return 0;          // target shrank — reset
        const lag = targetLen - prev;
        if (lag <= 0) { caughtUp = true; return prev; }
        const speedMultiplier = lag > maxLag ? 1 + (lag - maxLag) / maxLag : 1;
        const advance = Math.max(1, Math.round(charsPerSecond * speedMultiplier * dt));
        const next = Math.min(targetLen, prev + advance);
        if (next === targetLen) caughtUp = true;
        return next;
      });

      // Self-terminate only when the stream has ended AND we've revealed
      // everything. If the stream is still running, keep ticking. If we're
      // behind on a finished stream, keep draining.
      if (!enabledRef.current && caughtUp) {
        rafRef.current = null;
        lastTickRef.current = null;
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    // Start ticking when there's work to do: either the stream is active,
    // or it ended but we haven't caught up yet.
    if (enabled || displayedRef.current < target.length) {
      rafRef.current = requestAnimationFrame(tick);
    }

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      lastTickRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, charsPerSecond, maxLag]);

  // Always slice — even after enabled=false, the rAF loop above keeps
  // advancing `displayed` until it catches up, so this slice naturally
  // converges to the full target without ever snapping.
  return target.slice(0, displayed);
}
