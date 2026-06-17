"use client";

import { useCallback, useEffect, useState } from "react";

import {
  type BrainInfo,
  type FolderInfo,
  addBrainFolder,
  createBrain,
  deleteBrain,
  listBrains,
  listFolders,
  removeBrainFolder,
  renameBrain,
} from "@/lib/gateway";

/**
 * Manage user-defined brains: create / rename / delete a brain, and check the
 * vault folders that belong to it. A brain is a curated bucket of folders — a
 * doc is in the brain if its top-level folder is checked. Deleting a brain
 * never deletes documents.
 */
export function BrainsPanel({
  token,
  onChanged,
  onToast,
}: {
  token: string | null;
  onChanged: () => void;
  onToast: (msg: string) => void;
}) {
  const [brains, setBrains] = useState<BrainInfo[]>([]);
  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editId, setEditId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const [b, f] = await Promise.all([listBrains(token), listFolders(token)]);
      setBrains(b.brains);
      setFolders(f.folders);
    } catch (err) {
      onToast(`Failed to load brains: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setLoading(false);
    }
  }, [token, onToast]);

  useEffect(() => { void load(); }, [load]);

  // Reload locally AND tell the parent so the chat selector refreshes too.
  const refresh = useCallback(async () => {
    await load();
    onChanged();
  }, [load, onChanged]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!token || !name) return;
    setBusyId("__create__");
    try {
      await createBrain(token, name);
      setNewName("");
      setCreating(false);
      await refresh();
    } catch (err) {
      onToast(`Create failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  };

  const handleRename = async (id: string) => {
    const name = editName.trim();
    if (!token || !name) return;
    setBusyId(id);
    try {
      await renameBrain(token, id, name);
      setEditId(null);
      await refresh();
    } catch (err) {
      onToast(`Rename failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (b: BrainInfo) => {
    if (!token) return;
    if (!window.confirm(`Delete brain "${b.name}"? Your documents are NOT deleted.`)) return;
    setBusyId(b.id);
    try {
      await deleteBrain(token, b.id);
      await refresh();
    } catch (err) {
      onToast(`Delete failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  };

  const toggleFolder = async (b: BrainInfo, folder: string, isIn: boolean) => {
    if (!token) return;
    setBusyId(b.id);
    try {
      if (isIn) await removeBrainFolder(token, b.id, folder);
      else await addBrainFolder(token, b.id, folder);
      await refresh();
    } catch (err) {
      onToast(`Update failed: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setBusyId(null);
    }
  };

  const iconBtn =
    "text-[10px] px-1.5 py-0.5 rounded bg-white/[0.04] text-[#888] hover:bg-white/[0.08] hover:text-[#ccc] transition-all disabled:opacity-40";

  return (
    <div className="mt-2 pt-2 border-t border-white/[0.06]">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-[#aaa] font-medium tracking-wide">
          Brains {brains.length > 0 && <span className="text-[#666]">({brains.length})</span>}
        </span>
        <button
          onClick={() => setCreating((c) => !c)}
          className="text-[10px] px-2 py-0.5 rounded-lg bg-[#33ccff]/15 text-[#66ddff] hover:bg-[#33ccff]/25 transition-all"
        >
          {creating ? "Cancel" : "+ Brain"}
        </button>
      </div>

      {creating && (
        <div className="mb-2 space-y-1.5 p-2 rounded-lg bg-white/[0.03]">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void handleCreate(); }}
            placeholder="Brain name (e.g. Books, Me)"
            className="w-full text-[11px] px-2 py-1 rounded bg-white/[0.05] text-[#ccc] placeholder:text-[#555] outline-none border border-white/[0.06] focus:border-[#33ccff]/40"
          />
          <button
            onClick={() => void handleCreate()}
            disabled={!newName.trim() || busyId === "__create__"}
            className="w-full text-[10px] px-2 py-1 rounded bg-[#33ccff]/20 text-[#66ddff] hover:bg-[#33ccff]/30 transition-all disabled:opacity-40"
          >
            {busyId === "__create__" ? "Creating..." : "Create brain"}
          </button>
        </div>
      )}

      {loading && brains.length === 0 ? (
        <p className="text-[10px] text-[#666]">Loading…</p>
      ) : brains.length === 0 ? (
        <p className="text-[10px] text-[#666]">
          No brains yet. Create one, then check the folders it should cover to scope chat to it.
        </p>
      ) : (
        <div className="space-y-1.5">
          {brains.map((b) => {
            const inFolders = new Set(b.folders);
            return (
              <div key={b.id} className="rounded-lg bg-white/[0.02] p-1.5">
                <div className="flex items-center justify-between gap-2">
                  {editId === b.id ? (
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") void handleRename(b.id); }}
                      autoFocus
                      className="flex-1 text-[11px] px-2 py-1 rounded bg-white/[0.05] text-[#ccc] outline-none border border-white/[0.06] focus:border-[#33ccff]/40"
                    />
                  ) : (
                    <div className="min-w-0 flex-1">
                      <span className="text-[11px] text-[#ccc] truncate">{b.name}</span>
                      <span className="text-[9px] text-[#666] ml-1.5">
                        {b.doc_count} doc{b.doc_count === 1 ? "" : "s"} · {b.folders.length} folder{b.folders.length === 1 ? "" : "s"}
                      </span>
                      {b.missing_folders.length > 0 && (
                        <span
                          className="text-[9px] text-[#ffaa00] ml-1.5"
                          title={`No documents (renamed/deleted) — re-point: ${b.missing_folders.join(", ")}`}
                        >
                          ⚠ {b.missing_folders.length} missing
                        </span>
                      )}
                    </div>
                  )}
                  <div className="flex items-center gap-1 shrink-0">
                    {editId === b.id ? (
                      <button
                        onClick={() => void handleRename(b.id)}
                        disabled={busyId === b.id}
                        className="text-[10px] px-1.5 py-0.5 rounded bg-[#00ff41]/10 text-[#00ff41] hover:bg-[#00ff41]/20 transition-all disabled:opacity-40"
                      >
                        Save
                      </button>
                    ) : (
                      <>
                        <button
                          onClick={() => setExpandedId(expandedId === b.id ? null : b.id)}
                          title="Edit folders"
                          className={iconBtn}
                        >
                          Folders
                        </button>
                        <button
                          onClick={() => { setEditId(b.id); setEditName(b.name); }}
                          title="Rename"
                          className={iconBtn}
                        >
                          ✎
                        </button>
                        <button
                          onClick={() => void handleDelete(b)}
                          disabled={busyId === b.id}
                          title="Delete brain (documents are kept)"
                          className="text-[10px] px-1.5 py-0.5 rounded bg-white/[0.04] text-[#888] hover:bg-[#ff4444]/15 hover:text-[#ff6666] transition-all disabled:opacity-40"
                        >
                          ×
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {expandedId === b.id && (
                  <div className="mt-1.5 pt-1.5 border-t border-white/[0.05] space-y-1 max-h-40 overflow-y-auto">
                    {folders.length === 0 ? (
                      <p className="text-[9px] text-[#666]">No vault folders yet — sync a vault first.</p>
                    ) : (
                      folders.map((f) => {
                        const isIn = inFolders.has(f.folder);
                        return (
                          <label key={f.folder} className="flex items-center gap-1.5 text-[10px] text-[#aaa] cursor-pointer">
                            <input
                              type="checkbox"
                              checked={isIn}
                              disabled={busyId === b.id}
                              onChange={() => void toggleFolder(b, f.folder, isIn)}
                            />
                            <span className="truncate">{f.folder}</span>
                            <span className="text-[9px] text-[#666]">({f.doc_count})</span>
                          </label>
                        );
                      })
                    )}
                    {b.missing_folders.map((mf) => (
                      <label
                        key={`missing-${mf}`}
                        className="flex items-center gap-1.5 text-[10px] text-[#ffaa00] cursor-pointer"
                        title="This folder has no documents (renamed/deleted). Uncheck to re-point."
                      >
                        <input
                          type="checkbox"
                          checked
                          disabled={busyId === b.id}
                          onChange={() => void toggleFolder(b, mf, true)}
                        />
                        <span className="truncate">⚠ {mf}</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
