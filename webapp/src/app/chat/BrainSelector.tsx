"use client";

import { useEffect, useState } from "react";

import { type BrainInfo, listBrains } from "@/lib/gateway";

/**
 * Scope retrieval to a single user-defined "brain" (a named bucket of folders).
 * The value is a brain id; an empty value means "All brains" (no scope). The
 * list is fetched from GET /documents/brains; if the persisted brain no longer
 * exists (deleted) the selector silently falls back to All.
 */
export default function BrainSelector({
  token,
  selectedBrain,
  onChange,
  refreshKey,
}: {
  token: string | null;
  selectedBrain: string | null;
  onChange: (brain: string | null) => void;
  // Bumped by the parent when brains change in the manager panel, to refetch.
  refreshKey?: number;
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
  }, [token, refreshKey]);

  // If the saved brain disappeared (folder renamed/deleted), reset to All.
  useEffect(() => {
    if (
      selectedBrain &&
      brains.length > 0 &&
      !brains.some((b) => b.id === selectedBrain)
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
        <option key={b.id} value={b.id}>
          {b.name} ({b.doc_count})
        </option>
      ))}
    </select>
  );
}
