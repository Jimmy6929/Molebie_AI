"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { type SessionInfo } from "@/lib/gateway";

interface SidebarProps {
  sessions: SessionInfo[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
  onPinSession?: (id: string, pinned: boolean) => void;
  onLogout: () => void;
  userEmail: string;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onRenameSession,
  onPinSession,
  onLogout,
  userEmail,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId) {
      editInputRef.current?.focus();
      editInputRef.current?.select();
    }
  }, [editingId]);

  function timeAgo(dateStr: string): string {
    const now = new Date();
    const date = new Date(dateStr);
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  function startRename(s: SessionInfo) {
    setEditingId(s.id);
    setEditTitle(s.title);
  }

  function commitRename() {
    if (editingId && editTitle.trim()) {
      onRenameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle("");
  }

  function cancelRename() {
    setEditingId(null);
    setEditTitle("");
  }

  // Filter sessions by search query
  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions;
    const q = searchQuery.toLowerCase();
    return sessions.filter((s) => (s.title || "").toLowerCase().includes(q));
  }, [sessions, searchQuery]);

  // Sort: pinned first, then by updated_at (server already does this but ensure client-side)
  const sortedSessions = useMemo(() => {
    return [...filteredSessions].sort((a, b) => {
      const aPinned = (a as SessionInfo & { is_pinned?: boolean }).is_pinned ? 1 : 0;
      const bPinned = (b as SessionInfo & { is_pinned?: boolean }).is_pinned ? 1 : 0;
      if (bPinned !== aPinned) return bPinned - aPinned;
      return 0; // preserve server ordering otherwise
    });
  }, [filteredSessions]);

  const showSearch = sessions.length >= 5;

  return (
    <div className="w-64 flex flex-col min-h-screen bg-[#080808]/80 backdrop-blur-xl">
      {/* Header */}
      <div className="p-4 pb-3">
        <div className="text-[10px] text-[#999]">Local AI v0.1</div>
      </div>

      {/* New Chat */}
      <div className="px-3 pb-2">
        <button
          onClick={onNewChat}
          className="w-full text-left text-xs px-3 py-2.5 rounded-xl bg-[#00ff41]/[0.12] text-[#00ff41] hover:bg-[#00ff41]/[0.18] hover:text-[#33ff66] transition-all"
        >
          + New session
        </button>
      </div>

      {/* Search */}
      {showSearch && (
        <div className="px-3 pb-2">
          <div className="relative">
            <svg
              width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#666]"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search sessions..."
              className="w-full bg-white/[0.04] border border-white/[0.06] text-[#ccc] placeholder-[#555] pl-8 pr-7 py-1.5 text-[11px] font-mono rounded-lg focus:outline-none focus:border-[#00ff41]/30 transition-colors"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-[#666] hover:text-[#aaa] transition-colors"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto px-3 space-y-1 py-1">
        {sessions.length === 0 && (
          <div className="text-[10px] text-[#999] p-3 text-center">
            No sessions yet
          </div>
        )}
        {searchQuery && filteredSessions.length === 0 && sessions.length > 0 && (
          <div className="text-[10px] text-[#999] p-3 text-center">
            No matches
          </div>
        )}
        {sortedSessions.map((s) => {
          const isPinned = (s as SessionInfo & { is_pinned?: boolean }).is_pinned;
          return (
            <div
              key={s.id}
              className={`group flex items-center text-xs cursor-pointer transition-all rounded-xl ${
                activeSessionId === s.id
                  ? "bg-[#00ff41]/[0.12] text-[#00ff41]"
                  : "text-[#aaa] hover:text-[#eee] hover:bg-white/[0.05]"
              }`}
            >
              {editingId === s.id ? (
                <div className="flex-1 px-3 py-2">
                  <input
                    ref={editInputRef}
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename();
                      if (e.key === "Escape") cancelRename();
                    }}
                    onBlur={commitRename}
                    className="w-full bg-black/40 border border-[#00ff41]/30 text-[#00ff41] px-2 py-1 text-xs font-mono rounded-lg focus:outline-none focus:border-[#00ff41]/60"
                    maxLength={200}
                  />
                  <div className="text-[10px] text-[#888] mt-1">
                    Enter to save · Esc to cancel
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => onSelectSession(s.id)}
                  onDoubleClick={() => startRename(s)}
                  className="flex-1 text-left px-3 py-2.5 min-w-0"
                >
                  <div className="truncate flex items-center gap-1.5">
                    {isPinned && (
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" className="shrink-0 text-[#ffcc33]">
                        <path d="M16 2L14.5 3.5L18.5 7.5L20 6L16 2M12.5 5.5L8 10L9.5 11.5L5 16V19H8L12.5 14.5L14 16L18.5 11.5L12.5 5.5Z" />
                      </svg>
                    )}
                    {s.title || "Untitled"}
                  </div>
                  <div className="text-[10px] text-[#888] mt-0.5">
                    {timeAgo(s.updated_at)}
                  </div>
                </button>
              )}
              {editingId !== s.id && (
                <div className="flex opacity-0 group-hover:opacity-100 transition-opacity pr-1">
                  {onPinSession && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onPinSession(s.id, !isPinned);
                      }}
                      className={`p-1 rounded-lg hover:bg-white/[0.06] transition-all ${
                        isPinned ? "text-[#ffcc33] opacity-100" : "text-[#888] hover:text-[#ffcc33]"
                      }`}
                      title={isPinned ? "Unpin" : "Pin"}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill={isPinned ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M16 2L14.5 3.5L18.5 7.5L20 6L16 2M12.5 5.5L8 10L9.5 11.5L5 16V19H8L12.5 14.5L14 16L18.5 11.5L12.5 5.5Z" />
                      </svg>
                    </button>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startRename(s);
                    }}
                    className="p-1 text-[#888] hover:text-[#00ff41] rounded-lg hover:bg-white/[0.06] transition-all"
                    title="Rename"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                    </svg>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(s.id);
                    }}
                    className="p-1 text-[#888] hover:text-[#ff5555] rounded-lg hover:bg-[#ff5555]/[0.08] transition-all"
                    title="Delete"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 6h18" />
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* User Info */}
      <div className="p-4 space-y-2">
        <div className="text-[10px] text-[#999] truncate">
          {userEmail}
        </div>
        <button
          onClick={onLogout}
          className="text-[10px] text-[#aaa] hover:text-[#ff5555] transition-colors"
        >
          Sign out
        </button>
      </div>
    </div>
  );
}
