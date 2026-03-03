"use client";

import { useState, useRef, useEffect } from "react";
import { type SessionInfo } from "@/lib/gateway";

interface SidebarProps {
  sessions: SessionInfo[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
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
  onLogout,
  userEmail,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

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

  return (
    <div className="w-64 shrink-0 flex flex-col h-full glass rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 flex items-center justify-between">
        <div className="text-[11px] text-[#00ff41] glow font-bold tracking-wider">
          LOCAL_AI
        </div>
        <span className="text-[9px] text-[#555] bg-white/[0.05] px-1.5 py-0.5 rounded-full">v0.1</span>
      </div>

      {/* New Chat */}
      <div className="px-3 pt-3 pb-1">
        <button
          onClick={onNewChat}
          className="w-full text-[11px] px-3 py-2.5 rounded-xl border border-dashed border-white/[0.1] text-[#77bb88] hover:border-[#00ff41]/30 hover:text-[#00ff41] hover:bg-[#00ff41]/5 transition-all flex items-center gap-2"
        >
          <span className="text-sm">+</span> new session
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5 min-h-0">
        {sessions.length === 0 && (
          <div className="text-[10px] text-[#444] p-4 text-center">
            no sessions yet
          </div>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center text-[11px] cursor-pointer transition-all rounded-xl ${
              activeSessionId === s.id
                ? "bg-[#00ff41]/10 border border-[#00ff41]/15 text-[#00ff41]"
                : "border border-transparent text-[#888] hover:text-[#ccc] hover:bg-white/[0.04]"
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
                  className="w-full bg-black/30 border border-[#00ff41]/40 rounded-lg text-[#00ff41] px-2 py-1 text-[11px] font-mono focus:outline-none"
                  maxLength={200}
                />
                <div className="text-[9px] text-[#444] mt-1">
                  enter · esc
                </div>
              </div>
            ) : (
              <button
                onClick={() => onSelectSession(s.id)}
                onDoubleClick={() => startRename(s)}
                className="flex-1 text-left px-3 py-2.5 min-w-0"
              >
                <div className="truncate">{s.title || "untitled"}</div>
                <div className="text-[9px] text-[#555] mt-0.5">
                  {timeAgo(s.updated_at)}
                </div>
              </button>
            )}
            {editingId !== s.id && (
              <div className="flex shrink-0 opacity-0 group-hover:opacity-100 transition-opacity pr-2 gap-0.5">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    startRename(s);
                  }}
                  className="w-6 h-6 rounded-md flex items-center justify-center text-[10px] text-[#555] hover:text-[#00ff41] hover:bg-white/[0.05] transition-all"
                  title="rename session"
                >
                  ✎
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(s.id);
                  }}
                  className="w-6 h-6 rounded-md flex items-center justify-center text-[10px] text-[#555] hover:text-[#ff3333] hover:bg-[#ff3333]/10 transition-all"
                  title="delete session"
                >
                  ✕
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* User Info */}
      <div className="px-4 py-3 flex items-center justify-between">
        <div className="text-[10px] text-[#555] truncate min-w-0 flex-1">
          {userEmail}
        </div>
        <button
          onClick={onLogout}
          className="text-[10px] text-[#555] hover:text-[#ff3333] transition-all shrink-0 ml-2 px-2 py-1 rounded-md hover:bg-[#ff3333]/10"
        >
          logout
        </button>
      </div>
    </div>
  );
}
