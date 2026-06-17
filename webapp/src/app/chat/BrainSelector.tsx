"use client";

import { useEffect, useState } from "react";

import { type BrainInfo, listBrains } from "@/lib/gateway";

/**
 * Scope retrieval to a single "brain" (a top-level vault folder). An empty
 * value means "All brains" (no scope). The list is fetched from
 * GET /documents/brains; if the persisted brain no longer exists (folder
 * renamed/deleted) the selector silently falls back to All.
 */
export default function BrainSelector({
  token,
  selectedBrain,
  onChange,
}: {
  token: string | null;
  selectedBrain: string | null;
  onChange: (brain: string | null) => void;
}) {
  const [brains, setBrains] = useState<BrainInfo[]>([]);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    listBrains(token)
      .then((res) => {
        if (!cancelled) setBrains(res.brains);
      })
      .catch(() => {
        // Non-fatal: the selector just stays on "All Brains".
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  // If the saved brain disappeared (folder renamed/deleted), reset to All.
  useEffect(() => {
    if (
      selectedBrain &&
      brains.length > 0 &&
      !brains.some((b) => b.brain === selectedBrain)
    ) {
      onChange(null);
    }
  }, [brains, selectedBrain, onChange]);

  return (
    <select
      value={selectedBrain ?? ""}
      onChange={(e) => onChange(e.target.value || null)}
      title="Scope retrieval to one brain (top-level vault folder)"
      aria-label="Active brain"
      className="text-[11px] px-2 py-1.5 rounded-xl bg-white/[0.04] text-[#ccc] border border-white/[0.08] hover:border-white/[0.15] focus:outline-none focus:border-[#33ccff]/40 max-w-[150px] truncate cursor-pointer shrink-0"
    >
      <option value="">🧠 All Brains</option>
      {brains.map((b) => (
        <option key={b.brain} value={b.brain}>
          {b.brain} ({b.doc_count})
        </option>
      ))}
    </select>
  );
}
