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
import MessageBubble from "./MessageBubble";

interface DisplayMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  mode_used?: string | null;
  model_used?: string | null;
  streaming?: boolean;
  streamStartedAt?: number;
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
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const isStreamingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const userScrolledUpRef = useRef(false);

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
            role: m.role as DisplayMessage["role"],
            content: m.content,
            mode_used: m.mode_used,
            model_used: m.model_used ?? null,
          }))
        );
      } catch (err) {
        console.error("Failed to load messages:", err);
      }
    }
    loadMessages();
  }, [token, activeSessionId]);

  // Smart auto-scroll: only scroll down if user hasn't scrolled up
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Track whether the user has scrolled away from the bottom
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    function handleScroll() {
      const el = container!;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      userScrolledUpRef.current = !atBottom;
    }
    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  // Send message
  async function handleSend() {
    if (!input.trim() || !token || loading) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);
    userScrolledUpRef.current = false;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const now = Date.now();

    const userMsg: DisplayMessage = {
      id: `temp-${now}`,
      role: "user",
      content: userMessage,
    };

    const assistantMsg: DisplayMessage = {
      id: `stream-${now}`,
      role: "assistant",
      content: "",
      streaming: true,
      mode_used: mode,
      streamStartedAt: now,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    isStreamingRef.current = true;

    try {
      await sendMessageStream(
        token,
        userMessage,
        mode,
        activeSessionId || undefined,
        (content) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.streaming ? { ...m, content } : m
            )
          );
        },
        (sid) => {
          setActiveSessionId(sid);
        },
        controller.signal
      );

      setMessages((prev) =>
        prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
      );

      loadSessions();
    } catch (err) {
      if (controller.signal.aborted) {
        setMessages((prev) =>
          prev.map((m) =>
            m.streaming ? { ...m, streaming: false } : m
          )
        );
      } else {
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
      }
    } finally {
      abortControllerRef.current = null;
      isStreamingRef.current = false;
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  // Stop generation
  function handleStop() {
    abortControllerRef.current?.abort();
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
      <div className="min-h-screen flex items-center justify-center">
        <div className="glow">authenticating...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex">
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
      <div className="flex-1 flex flex-col min-h-screen">
        {/* Header */}
        <header className="border-b border-[#3a3a3a] px-4 py-2 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-[#77bb88] hover:text-[#00ff41] text-sm"
            >
              {sidebarOpen ? "[≡]" : "[≡]"}
            </button>
            <span className="text-xs text-[#77bb88]">
              {activeSessionId
                ? `session:${activeSessionId.slice(0, 8)}...`
                : "new_session"}
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Mode toggle */}
            <button
              onClick={() =>
                setMode(mode === "instant" ? "thinking" : "instant")
              }
              className={`text-xs px-2 py-1 border transition-colors ${
                mode === "thinking"
                  ? "border-[#ff9900] text-[#ff9900] bg-[#ff9900]/10"
                  : "border-[#3a3a3a] text-[#77bb88] hover:border-[#77bb88]"
              }`}
            >
              {mode === "thinking" ? "[think]" : "[instant]"}
            </button>
          </div>
        </header>

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-[#555555] space-y-2">
                <pre className="text-xs leading-tight">
{`
  ┌─────────────────────────┐
  │  ready for input...     │
  │                         │
  │  type a message below   │
  │  to start a session     │
  └─────────────────────────┘
`}
                </pre>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              role={msg.role}
              content={msg.content}
              streaming={msg.streaming}
              mode={msg.mode_used}
              model={msg.model_used}
              streamStartedAt={msg.streamStartedAt}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-[#3a3a3a] p-4 shrink-0">
          <div className="flex gap-2 items-end max-w-4xl mx-auto">
            <div className="text-[#77bb88] text-sm pt-2">{">"}</div>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="type your message..."
              rows={1}
              className="flex-1 bg-transparent text-[#00ff41] text-sm resize-none focus:outline-none placeholder-[#555555] min-h-[36px] max-h-[200px] py-2"
              style={{
                height: "auto",
                overflow: "hidden",
              }}
              onInput={(e) => {
                const t = e.target as HTMLTextAreaElement;
                t.style.height = "auto";
                t.style.height = t.scrollHeight + "px";
              }}
              disabled={loading}
              autoFocus
            />
            {loading ? (
              <button
                onClick={handleStop}
                className="text-sm px-3 py-2 border border-[#ff3333]/40 text-[#ff3333] hover:border-[#ff3333] hover:bg-[#ff3333]/10 transition-colors shrink-0"
              >
                [stop]
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="text-sm px-3 py-2 border border-[#3a3a3a] text-[#77bb88] hover:border-[#00ff41] hover:text-[#00ff41] transition-colors disabled:opacity-30 disabled:cursor-not-allowed shrink-0"
              >
                [send]
              </button>
            )}
          </div>
          <div className="text-[10px] text-[#555555] mt-1 text-center">
            enter to send · shift+enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}
