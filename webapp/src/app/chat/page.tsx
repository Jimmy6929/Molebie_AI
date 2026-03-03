"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase";
import {
  sendMessageStream,
  listSessions,
  getSessionMessages,
  deleteSession,
  renameSession,
  type SessionInfo,
  type ChatMessage,
} from "@/lib/gateway";
import Sidebar from "./sidebar";

interface DisplayMessage {
  id: string;
  role: string;
  content: string;
  mode_used?: string | null;
  streaming?: boolean;
}

export default function ChatPage() {
  const router = useRouter();
  const [token, setToken] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string>("");
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"instant" | "thinking">("instant");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const isStreamingRef = useRef(false);

  // Auth check
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.replace("/login");
        return;
      }
      setToken(session.access_token);
      setUserEmail(session.user.email || "");
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) {
        router.replace("/login");
      } else {
        setToken(session.access_token);
      }
    });

    return () => subscription.unsubscribe();
  }, [router]);

  // Load sessions
  const loadSessions = useCallback(async () => {
    if (!token) return;
    try {
      const data = await listSessions(token);
      setSessions(data.sessions);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    }
  }, [token]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Load messages when active session changes
  useEffect(() => {
    if (!token || !activeSessionId) {
      setMessages([]);
      return;
    }

    // Don't reload from DB while streaming — the stream handler manages state
    if (isStreamingRef.current) return;

    async function loadMessages() {
      try {
        const msgs = await getSessionMessages(token!, activeSessionId!);
        setMessages(
          msgs.map((m: ChatMessage) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            mode_used: m.mode_used,
          }))
        );
      } catch (err) {
        console.error("Failed to load messages:", err);
      }
    }
    loadMessages();
  }, [token, activeSessionId]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Send message
  async function handleSend() {
    if (!input.trim() || !token || loading) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);

    // Add user message to display
    const userMsg: DisplayMessage = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: userMessage,
    };

    // Add streaming placeholder
    const assistantMsg: DisplayMessage = {
      id: `stream-${Date.now()}`,
      role: "assistant",
      content: "",
      streaming: true,
      mode_used: mode,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    isStreamingRef.current = true;

    try {
      await sendMessageStream(
        token,
        userMessage,
        mode,
        activeSessionId || undefined,
        // onChunk
        (content) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.streaming ? { ...m, content } : m
            )
          );
        },
        // onSessionId
        (sid) => {
          setActiveSessionId(sid);
        }
      );

      // Mark as done streaming
      setMessages((prev) =>
        prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
      );

      // Refresh sessions
      loadSessions();
    } catch (err) {
      console.error("Send error:", err);
      setMessages((prev) =>
        prev.map((m) =>
          m.streaming
            ? {
                ...m,
                content: `error: ${err instanceof Error ? err.message : "connection failed"}`,
                streaming: false,
              }
            : m
        )
      );
    } finally {
      isStreamingRef.current = false;
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  // New chat
  function handleNewChat() {
    setActiveSessionId(null);
    setMessages([]);
    inputRef.current?.focus();
  }

  // Delete session
  async function handleDeleteSession(id: string) {
    if (!token) return;
    try {
      await deleteSession(token, id);
      if (activeSessionId === id) {
        setActiveSessionId(null);
        setMessages([]);
      }
      loadSessions();
    } catch (err) {
      console.error("Delete error:", err);
    }
  }

  // Rename session
  async function handleRenameSession(id: string, title: string) {
    if (!token) return;
    try {
      await renameSession(token, id, title);
      loadSessions();
    } catch (err) {
      console.error("Rename error:", err);
    }
  }

  // Logout
  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace("/login");
  }

  // Handle key press
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  if (!token) {
    return (
      <div className="h-screen flex items-center justify-center bg-[#050505]">
        <div className="glow text-sm">authenticating...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex overflow-hidden bg-[#050505] p-2 gap-2">
      {/* Sidebar */}
      {sidebarOpen && (
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={setActiveSessionId}
          onNewChat={handleNewChat}
          onDeleteSession={handleDeleteSession}
          onRenameSession={handleRenameSession}
          onLogout={handleLogout}
          userEmail={userEmail}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full min-w-0 glass rounded-2xl overflow-hidden relative">
        {/* Floating Header Island */}
        <div className="absolute top-3 left-0 right-0 z-10 flex justify-center px-5 pointer-events-none">
          <header className="pointer-events-auto inline-flex items-center gap-3 px-1.5 py-1.5 rounded-full bg-black/50 backdrop-blur-2xl border border-white/[0.08] shadow-lg shadow-black/20">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-7 h-7 rounded-full flex items-center justify-center text-[#77bb88] hover:text-[#00ff41] hover:bg-white/[0.08] text-xs transition-all"
            >
              ≡
            </button>
            <span className="text-[10px] text-[#555] font-normal px-1">
              {activeSessionId
                ? `${activeSessionId.slice(0, 8)}…`
                : "new session"}
            </span>
            <button
              onClick={() =>
                setMode(mode === "instant" ? "thinking" : "instant")
              }
              className={`text-[10px] px-2.5 py-1 rounded-full transition-all ${
                mode === "thinking"
                  ? "text-[#ff9900] bg-[#ff9900]/15"
                  : "text-[#77bb88] bg-white/[0.06] hover:bg-white/[0.1]"
              }`}
            >
              {mode === "thinking" ? "◆ think" : "⚡ instant"}
            </button>
          </header>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto min-h-0 px-5 pt-16 pb-40">
          <div className="max-w-3xl mx-auto space-y-3">
            {messages.length === 0 && (
              <div className="flex items-center justify-center" style={{ minHeight: 'calc(100vh - 200px)' }}>
                <div className="text-center space-y-3">
                  <div className="text-3xl opacity-20">⌘</div>
                  <div className="text-xs text-[#555]">ready for input…</div>
                  <div className="text-[10px] text-[#333]">type a message below to start a session</div>
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] px-4 py-3 text-sm transition-all ${
                    msg.role === "user"
                      ? "rounded-2xl rounded-br-md bg-[#00ff41]/10 border border-[#00ff41]/15 text-[#00ff41]"
                      : "rounded-2xl rounded-bl-md bg-white/[0.04] border border-white/[0.06] text-[#b0b0b0]"
                  }`}
                >
                  <div className="text-[10px] mb-1.5 select-none flex items-center gap-1.5">
                    <span className={msg.role === "user" ? "text-[#00ff41]/50" : "text-[#555]"}>
                      {msg.role === "user" ? "> you" : "> ai"}
                    </span>
                    {msg.mode_used === "thinking" && (
                      <span className="text-[#ff9900]/70 text-[9px] bg-[#ff9900]/10 px-1.5 py-0.5 rounded-full">think</span>
                    )}
                  </div>
                  <div className="whitespace-pre-wrap break-words overflow-hidden leading-relaxed">
                    {msg.content}
                    {msg.streaming && (
                      <span className="inline-block w-1.5 h-4 bg-[#00ff41] ml-0.5 animate-pulse rounded-full" />
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Floating Input Island */}
        <div className="absolute bottom-3 left-0 right-0 z-10 px-5 pointer-events-none">
          <div className="max-w-3xl mx-auto pointer-events-auto">
            <div className="flex gap-2 items-end rounded-2xl bg-black/50 backdrop-blur-2xl border border-white/[0.08] shadow-lg shadow-black/20 px-4 py-2.5 focus-within:border-[#00ff41]/20 transition-all">
              <div className="text-[#555] text-sm pt-0.5 select-none shrink-0">{">"}</div>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  const t = e.target as HTMLTextAreaElement;
                  t.style.height = "0px";
                  t.style.height = Math.min(t.scrollHeight, 200) + "px";
                }}
                onKeyDown={handleKeyDown}
                placeholder="type your message…"
                rows={1}
                className="flex-1 bg-transparent text-[#00ff41] text-sm resize-none focus:outline-none placeholder-[#333] py-0.5 min-h-[24px] max-h-[200px] overflow-y-auto"
                disabled={loading}
                autoFocus
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="text-[11px] px-3.5 py-1.5 rounded-full bg-[#00ff41]/10 border border-[#00ff41]/20 text-[#00ff41] hover:bg-[#00ff41]/20 hover:border-[#00ff41]/30 transition-all disabled:opacity-20 disabled:cursor-not-allowed shrink-0"
              >
                {loading ? "···" : "send ↵"}
              </button>
            </div>
            <div className="text-[10px] text-[#333] mt-1.5 text-center">
              enter to send · shift+enter for new line
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
