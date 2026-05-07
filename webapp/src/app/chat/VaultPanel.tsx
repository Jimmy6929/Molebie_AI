"use client";

import { useCallback, useEffect, useState } from "react";

import {
  connectVault,
  disconnectVault,
  listVaults,
  triggerVaultSync,
  type SyncVaultResponse,
  type VaultInfo,
} from "@/lib/gateway";

interface Props {
  token: string | null;
  onSyncJobStarted: (params: {
    jobId: string;
    rootLabel: string;
    totalFiles: number;
  }) => void;
  onDocumentsChanged: () => void;
  onToast: (msg: string) => void;
}

function formatLastSync(iso: string | null): string {
  if (!iso) return "never synced";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const ageSec = Math.floor((Date.now() - t) / 1000);
  if (ageSec < 60) return `synced ${ageSec}s ago`;
  if (ageSec < 3600) return `synced ${Math.floor(ageSec / 60)}m ago`;
  if (ageSec < 86400) return `synced ${Math.floor(ageSec / 3600)}h ago`;
  return `synced ${Math.floor(ageSec / 86400)}d ago`;
}

function summarizeClassification(r: SyncVaultResponse): string {
  const parts: string[] = [];
  if (r.classification.new) parts.push(`+${r.classification.new} new`);
  if (r.classification.adopted) parts.push(`↻${r.classification.adopted} adopted`);
  if (r.classification.changed) parts.push(`~${r.classification.changed} changed`);
  if (r.classification.deleted) parts.push(`−${r.classification.deleted} deleted`);
  if (r.classification.skipped) parts.push(`!${r.classification.skipped} skipped`);
  if (parts.length === 0) return `nothing changed (${r.classification.unchanged} unchanged)`;
  return parts.join(", ");
}

export function VaultPanel({
  token,
  onSyncJobStarted,
  onDocumentsChanged,
  onToast,
}: Props) {
  const [vaults, setVaults] = useState<VaultInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [label, setLabel] = useState("");
  const [rootPath, setRootPath] = useState("");
  const [indexAttachments, setIndexAttachments] = useState(true);

  const loadVaults = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const res = await listVaults(token);
      setVaults(res.vaults);
    } catch (err) {
      onToast(`Failed to load vaults: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setLoading(false);
    }
  }, [token, onToast]);

  useEffect(() => {
    void loadVaults();
  }, [loadVaults]);

  async function handleConnect() {
    if (!token) return;
    const trimmedLabel = label.trim();
    const trimmedPath = rootPath.trim();
    if (!trimmedLabel || !trimmedPath) {
      onToast("Vault label and path are required");
      return;
    }
    setBusyId("__connect__");
    try {
      await connectVault(token, {
        label: trimmedLabel,
        root_path: trimmedPath,
        index_attachments: indexAttachments,
      });
      setLabel("");
      setRootPath("");
      setAddOpen(false);
      await loadVaults();
    } catch (err) {
      onToast(`Connect failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  }

  async function handleSync(v: VaultInfo) {
    if (!token) return;
    setBusyId(v.id);
    try {
      const res = await triggerVaultSync(token, v.id);
      const dispatched = res.classification.new + res.classification.changed;
      if (res.job_id && dispatched > 0) {
        onSyncJobStarted({
          jobId: res.job_id,
          rootLabel: `vault:${v.label}`,
          totalFiles: dispatched,
        });
      }
      onToast(`${v.label}: ${summarizeClassification(res)}`);
      await loadVaults();
      onDocumentsChanged();
    } catch (err) {
      onToast(`Sync failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  }

  async function handleDisconnect(v: VaultInfo) {
    if (!token) return;
    const ok = window.confirm(
      `Disconnect "${v.label}"? This removes ${v.doc_count} document(s) from your Brain. ` +
      "Your vault files on disk are NOT touched.",
    );
    if (!ok) return;
    setBusyId(v.id);
    try {
      const res = await disconnectVault(token, v.id);
      onToast(`Removed ${res.deleted_documents} document(s) from "${v.label}"`);
      await loadVaults();
      onDocumentsChanged();
    } catch (err) {
      onToast(`Disconnect failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="mt-2 pt-2 border-t border-white/[0.06]">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] text-[#aaa] font-medium tracking-wide">
          Vaults {vaults.length > 0 && <span className="text-[#666]">({vaults.length})</span>}
        </span>
        <button
          onClick={() => setAddOpen((o) => !o)}
          className="text-[10px] px-2 py-0.5 rounded-lg bg-[#33ccff]/15 text-[#66ddff] hover:bg-[#33ccff]/25 transition-all"
          title="Connect an Obsidian vault (or any folder) for live-syncing RAG"
        >
          {addOpen ? "Cancel" : "+ Vault"}
        </button>
      </div>

      {addOpen && (
        <div className="mb-2 space-y-1.5 p-2 rounded-lg bg-white/[0.03]">
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Vault label (e.g. Personal Notes)"
            className="w-full text-[11px] px-2 py-1 rounded bg-white/[0.05] text-[#ccc] placeholder:text-[#555] outline-none border border-white/[0.06] focus:border-[#33ccff]/40"
          />
          <input
            type="text"
            value={rootPath}
            onChange={(e) => setRootPath(e.target.value)}
            placeholder="Absolute path (e.g. /Users/you/Documents/MyVault)"
            className="w-full text-[11px] px-2 py-1 rounded bg-white/[0.05] text-[#ccc] placeholder:text-[#555] outline-none border border-white/[0.06] focus:border-[#33ccff]/40 font-mono"
            spellCheck={false}
          />
          <label className="flex items-center gap-1.5 text-[10px] text-[#aaa]">
            <input
              type="checkbox"
              checked={indexAttachments}
              onChange={(e) => setIndexAttachments(e.target.checked)}
            />
            Also index PDFs and .txt
          </label>
          <button
            onClick={handleConnect}
            disabled={busyId === "__connect__" || !label.trim() || !rootPath.trim()}
            className="w-full text-[10px] px-2 py-1 rounded bg-[#33ccff]/20 text-[#66ddff] hover:bg-[#33ccff]/30 transition-all disabled:opacity-40"
          >
            {busyId === "__connect__" ? "Connecting..." : "Connect Vault"}
          </button>
        </div>
      )}

      {loading && vaults.length === 0 ? (
        <p className="text-[10px] text-[#666]">Loading vaults...</p>
      ) : vaults.length === 0 ? (
        <p className="text-[10px] text-[#666]">
          No vaults connected. Click + Vault to point Molebie at an Obsidian folder for live-sync RAG.
        </p>
      ) : (
        <div className="space-y-1.5">
          {vaults.map((v) => {
            const isBusy = busyId === v.id;
            return (
              <div
                key={v.id}
                className="flex items-center justify-between gap-2 group"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-[#33ccff]" />
                    <span className="text-[11px] text-[#ccc] truncate font-medium">
                      {v.label}
                    </span>
                    <span className="text-[9px] text-[#666] shrink-0">
                      {v.doc_count} doc{v.doc_count === 1 ? "" : "s"}
                    </span>
                  </div>
                  <div className="text-[9px] text-[#666] truncate font-mono ml-3">
                    {v.root_path}
                  </div>
                  <div className="text-[9px] text-[#888] ml-3">
                    {formatLastSync(v.last_sync_at)}
                  </div>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    onClick={() => handleSync(v)}
                    disabled={isBusy}
                    className="text-[10px] px-2 py-0.5 rounded bg-[#00ff41]/10 text-[#00ff41] hover:bg-[#00ff41]/20 transition-all disabled:opacity-40"
                  >
                    {isBusy ? "Syncing..." : "Sync"}
                  </button>
                  <button
                    onClick={() => handleDisconnect(v)}
                    disabled={isBusy}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-white/[0.04] text-[#888] hover:bg-[#ff4444]/15 hover:text-[#ff6666] transition-all disabled:opacity-40"
                    title="Disconnect vault"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
