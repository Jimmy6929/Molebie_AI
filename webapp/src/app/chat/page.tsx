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
import {
  useSpeechRecognition,
  useKokoroTTS,
  useVoiceSettings,
  isStopCommand,
  extractWakeCommand,
} from "@/lib/voice";
import Sidebar from "./sidebar";
import MessageBubble from "./MessageBubble";
import VoiceSettings from "./VoiceSettings";

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
  const [mode, setMode] = useState<"instant" | "thinking" | "thinking_harder">(
    "thinking"
  );
  const [conversationMode, setConversationMode] = useState(false);
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false);
  const [speakerVerifyEnabled, setSpeakerVerifyEnabled] = useState(false);
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [showImageToast, setShowImageToast] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isStreamingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const userScrolledUpRef = useRef(false);
  const sendVoiceMessageRef = useRef<(text: string) => void>(() => {});
  const continuousRef = useRef(false);

  const { settings: voiceSettings, setSettings: setVoiceSettings } =
    useVoiceSettings();
  const { isSpeaking, speak: kokoroSpeak, cancel: kokoroCancel } =
    useKokoroTTS();

  const onFinalTranscript = useCallback(
    (text: string) => {
      const clean = text.trim();
      if (!clean) return;

      if (isStopCommand(clean)) {
        continuousRef.current = false;
        setConversationMode(false);
        return;
      }

      if (wakeWordEnabled) {
        const { isWakeWord, command } = extractWakeCommand(clean);
        if (!isWakeWord) return; // not a wake word — ignore, loop restarts
        if (command) {
          setInput(command);
          sendVoiceMessageRef.current(command);
        }
        return;
      }

      setInput(clean);
      if (!conversationMode) return;
      sendVoiceMessageRef.current(clean);
    },
    [conversationMode, wakeWordEnabled]
  );

  const {
    supportsSpeechRecognition,
    isListening,
    isTranscribing,
    error: voiceInputError,
    startListening,
    stopListening,
    abortListening,
  } = useSpeechRecognition({
    token,
    onFinalTranscript,
    autoStopOnSilence: conversationMode,
    verifySpeaker: speakerVerifyEnabled,
  });

  // Auto-restart listening for continuous / wake-word mode
  useEffect(() => {
    if (!conversationMode) return;
    if (isListening || isTranscribing || isSpeaking || loading) return;
    if (!continuousRef.current && !wakeWordEnabled) return;

    const timer = setTimeout(() => {
      void startListening();
    }, 400);
    return () => clearTimeout(timer);
  }, [
    conversationMode,
    isListening,
    isTranscribing,
    isSpeaking,
    loading,
    wakeWordEnabled,
    startListening,
  ]);

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

  useEffect(() => {
    if (!token || !activeSessionId) {
      setMessages([]);
      return;
    }
    if (isStreamingRef.current) return;

    async function loadMessages() {
      try {
        const msgs = await getSessionMessages(token!, activeSessionId!);
        setMessages(
          msgs.map((m: ChatMessage) => {
            let content = m.content;
            if (m.reasoning_content) {
              content = `<think>${m.reasoning_content}</think>${content}`;
            }
            return {
              id: m.id,
              role: m.role as DisplayMessage["role"],
              content,
              mode_used: m.mode_used,
              model_used: m.model_used ?? null,
            };
          })
        );
      } catch (err) {
        console.error("Failed to load messages:", err);
      }
    }
    loadMessages();
  }, [token, activeSessionId]);

  useEffect(() => {
    if (userScrolledUpRef.current) return;
    const el = messagesContainerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    function checkPosition() {
      requestAnimationFrame(() => {
        const el = container!;
        const atBottom =
          el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        userScrolledUpRef.current = !atBottom;
        setShowScrollBtn(!atBottom);
      });
    }

    container.addEventListener("wheel", checkPosition, { passive: true });
    container.addEventListener("touchmove", checkPosition, { passive: true });
    return () => {
      container.removeEventListener("wheel", checkPosition);
      container.removeEventListener("touchmove", checkPosition);
    };
  }, []);

  function scrollToBottom() {
    userScrolledUpRef.current = false;
    setShowScrollBtn(false);
    const el = messagesContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }

  const resizeTextarea = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxH = 200;
    if (el.scrollHeight > maxH) {
      el.style.height = maxH + "px";
      el.style.overflowY = "auto";
    } else {
      el.style.height = el.scrollHeight + "px";
      el.style.overflowY = "hidden";
    }
  }, []);

  useEffect(() => {
    resizeTextarea();
  }, [input, resizeTextarea]);

  async function sendMessageWithText(rawInput: string) {
    if (!rawInput.trim() || !token || loading) return;

    const userMessage = rawInput.trim();
    setInput("");
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.overflowY = "hidden";
    }
    setLoading(true);
    userScrolledUpRef.current = false;
    setShowScrollBtn(false);

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
      mode_used: conversationMode ? "instant" : mode,
      streamStartedAt: now,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    isStreamingRef.current = true;

    try {
      if (conversationMode) {
        kokoroCancel();
      }

      const finalContent = await sendMessageStream(
        token,
        userMessage,
        conversationMode ? "instant" : mode,
        activeSessionId || undefined,
        conversationMode,
        (content) => {
          setMessages((prev) =>
            prev.map((m) => (m.streaming ? { ...m, content } : m))
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

      if (conversationMode && finalContent.trim() && token) {
        await kokoroSpeak(
          token,
          finalContent,
          voiceSettings.voiceId,
          voiceSettings.speed
        );
        // After TTS, auto-restart is handled by the useEffect above
      }

      loadSessions();
    } catch (err) {
      if (controller.signal.aborted) {
        setMessages((prev) =>
          prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
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

  sendVoiceMessageRef.current = (text: string) => {
    void sendMessageWithText(text);
  };

  function handleSend() {
    void sendMessageWithText(input);
  }

  function handleStop() {
    abortControllerRef.current?.abort();
    kokoroCancel();
    continuousRef.current = false;
  }

  function handleNewChat() {
    setActiveSessionId(null);
    setMessages([]);
    inputRef.current?.focus();
  }

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

  async function handleRenameSession(id: string, title: string) {
    if (!token) return;
    try {
      await renameSession(token, id, title);
      loadSessions();
    } catch (err) {
      console.error("Rename error:", err);
    }
  }

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace("/login");
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleMicClick() {
    if (!supportsSpeechRecognition || loading || isTranscribing || isSpeaking)
      return;
    if (isListening) {
      stopListening();
      return;
    }
    kokoroCancel();
    continuousRef.current = conversationMode;
    void startListening();
  }

  function handleEndConversation() {
    continuousRef.current = false;
    setConversationMode(false);
    kokoroCancel();
    abortListening();
    setVoiceSettingsOpen(false);
  }

  useEffect(() => {
    if (!conversationMode) {
      continuousRef.current = false;
      kokoroCancel();
      abortListening();
      setVoiceSettingsOpen(false);
      return;
    }
    setTimeout(() => inputRef.current?.focus(), 0);
  }, [abortListening, conversationMode, kokoroCancel]);

  function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
    setShowImageToast(true);
    setTimeout(() => setShowImageToast(false), 3000);
  }

  function removeImage() {
    setImageFile(null);
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glow text-sm">authenticating...</div>
      </div>
    );
  }

  const isConversationActive =
    conversationMode &&
    (isListening || isTranscribing || isSpeaking || loading);

  function statusText(): string {
    if (!conversationMode) return "Enter to send · Shift+Enter for new line";
    if (isSpeaking) return "Alfred is speaking...";
    if (isTranscribing) return "Transcribing your speech...";
    if (isListening) return "Listening... (auto-stops on silence)";
    if (loading) return "Thinking...";
    if (wakeWordEnabled)
      return 'Say "Hey Alfred" to start · say "stop" to end';
    return 'Voice mode · tap mic to start · say "stop" to end';
  }

  return (
    <div className="min-h-screen flex bg-[#0a0a0a]">
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
      <div className="flex-1 flex flex-col min-h-screen relative">
        {/* Header */}
        <header className="px-5 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-[#00ff41] hover:text-[#33ff66] transition-colors p-2 rounded-xl hover:bg-white/[0.06]"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <span className="text-xs text-[#999]">
              {activeSessionId
                ? `${activeSessionId.slice(0, 8)}...`
                : "New session"}
            </span>
          </div>
        </header>

        {/* Messages */}
        <div
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto px-4 py-4 space-y-3"
        >
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-[#999] text-sm">
                What can I help you with?
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

        {/* Scroll to bottom button */}
        {showScrollBtn && (
          <div className="absolute bottom-32 left-1/2 -translate-x-1/2 z-10 animate-fade-in-up">
            <button
              onClick={scrollToBottom}
              className="glass rounded-full p-2.5 text-[#00ff41] hover:text-[#33ff66] transition-all hover:scale-105 shadow-lg shadow-black/40"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          </div>
        )}

        {/* Image toast */}
        {showImageToast && (
          <div className="absolute bottom-36 left-1/2 -translate-x-1/2 z-20 animate-fade-in-up">
            <div className="glass rounded-xl px-4 py-2 text-xs text-[#ffcc33]">
              Image upload not yet connected to backend
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 pb-5 shrink-0">
          <div className="max-w-3xl mx-auto">
            <VoiceSettings
              open={voiceSettingsOpen}
              token={token}
              settings={voiceSettings}
              onChange={setVoiceSettings}
              wakeWordEnabled={wakeWordEnabled}
              onWakeWordToggle={setWakeWordEnabled}
              speakerVerifyEnabled={speakerVerifyEnabled}
              onSpeakerVerifyToggle={setSpeakerVerifyEnabled}
            />

            {/* Image preview */}
            {imagePreview && (
              <div className="mb-2 flex items-center gap-2 animate-fade-in">
                <div className="relative group">
                  <img
                    src={imagePreview}
                    alt="Selected"
                    className="h-16 w-16 object-cover rounded-xl border border-white/[0.06]"
                  />
                  <button
                    onClick={removeImage}
                    className="absolute -top-1.5 -right-1.5 bg-[#ff3333] text-white rounded-full w-5 h-5 flex items-center justify-center text-[10px] opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    ×
                  </button>
                </div>
                <span className="text-[10px] text-[#999]">
                  {imageFile?.name}
                </span>
              </div>
            )}

            <div className="glass rounded-2xl p-2 flex items-end gap-2 transition-all">
              {/* Image upload button */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 rounded-xl text-[#999] hover:text-[#00ff41] hover:bg-white/[0.06] transition-all shrink-0"
                title="Add image"
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="4" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                </svg>
              </button>

              {/* Mode toggle pill */}
              <button
                onClick={() =>
                  setMode(mode === "instant" ? "thinking" : "instant")
                }
                className={`text-[11px] px-3 py-1.5 rounded-xl transition-all shrink-0 ${
                  mode !== "instant"
                    ? "bg-[#ffbb33]/15 text-[#ffcc33] border border-[#ffbb33]/30"
                    : "text-[#999] hover:text-[#00ff41] border border-white/[0.08] hover:border-white/[0.15]"
                }`}
              >
                {mode === "instant"
                  ? "Fast"
                  : mode === "thinking_harder"
                    ? "Think+"
                    : "Think"}
              </button>

              {/* Think harder sub-toggle */}
              {mode !== "instant" && (
                <button
                  onClick={() =>
                    setMode(
                      mode === "thinking_harder" ? "thinking" : "thinking_harder"
                    )
                  }
                  className={`text-[11px] px-3 py-1.5 rounded-xl transition-all shrink-0 ${
                    mode === "thinking_harder"
                      ? "bg-[#ff8833]/20 text-[#ff9955] border border-[#ff8833]/40"
                      : "text-[#ffcc33] border border-[#ffbb33]/30 hover:bg-[#ffbb33]/10"
                  }`}
                >
                  {mode === "thinking_harder" ? "Think Harder" : "Normal"}
                </button>
              )}

              {/* Conversation mode toggle */}
              <button
                onClick={() => {
                  if (conversationMode) {
                    handleEndConversation();
                  } else {
                    setConversationMode(true);
                    continuousRef.current = true;
                  }
                }}
                className={`text-[11px] px-3 py-1.5 rounded-xl transition-all shrink-0 ${
                  conversationMode
                    ? "bg-[#33ccff]/20 text-[#66ddff] border border-[#33ccff]/40"
                    : "text-[#999] border border-white/[0.08] hover:text-[#66ddff] hover:border-white/[0.15]"
                }`}
                title="Voice conversation mode"
              >
                Voice
              </button>

              {conversationMode && (
                <>
                  <button
                    onClick={handleMicClick}
                    disabled={
                      !supportsSpeechRecognition ||
                      loading ||
                      isTranscribing ||
                      isSpeaking
                    }
                    className={`p-2 rounded-xl transition-all shrink-0 ${
                      isListening
                        ? "bg-[#ff4444]/20 text-[#ff6666] animate-pulse"
                        : isTranscribing
                          ? "bg-[#ffbb33]/15 text-[#ffcc33] animate-pulse"
                          : isSpeaking
                            ? "bg-[#33ccff]/15 text-[#66ddff] animate-pulse"
                            : "text-[#66ddff] hover:bg-[#33ccff]/10"
                    } disabled:opacity-30 disabled:cursor-not-allowed`}
                    title={
                      isSpeaking
                        ? "AI is speaking..."
                        : isTranscribing
                          ? "Transcribing..."
                          : isListening
                            ? "Stop recording"
                            : "Start conversation"
                    }
                  >
                    {isTranscribing ? (
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 6v6l4 2" />
                      </svg>
                    ) : isSpeaking ? (
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                        <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                      </svg>
                    ) : (
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                        <line x1="12" y1="19" x2="12" y2="23" />
                        <line x1="8" y1="23" x2="16" y2="23" />
                      </svg>
                    )}
                  </button>

                  {isConversationActive && (
                    <button
                      onClick={handleEndConversation}
                      className="p-2 rounded-xl bg-[#ff4444]/15 text-[#ff6666] hover:bg-[#ff4444]/25 transition-all shrink-0"
                      title="End conversation"
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <rect x="6" y="6" width="12" height="12" rx="2" />
                      </svg>
                    </button>
                  )}

                  <button
                    onClick={() => setVoiceSettingsOpen((prev) => !prev)}
                    className="p-2 rounded-xl text-[#66ddff] hover:bg-[#33ccff]/10 transition-all shrink-0"
                    title="Voice settings"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="3" />
                      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                    </svg>
                  </button>
                </>
              )}

              {/* Text input */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  resizeTextarea();
                }}
                onKeyDown={handleKeyDown}
                placeholder="Send a message..."
                rows={1}
                className="flex-1 bg-transparent text-[#f0f0f0] text-sm resize-none focus:outline-none placeholder-[#888] min-h-[36px] max-h-[200px] py-2 px-1"
                style={{ overflowY: "hidden" }}
                disabled={loading}
                autoFocus
              />

              {/* Send / Stop button */}
              {loading ? (
                <button
                  onClick={handleStop}
                  className="p-2 rounded-xl bg-[#ff5555]/15 text-[#ff5555] hover:bg-[#ff5555]/25 transition-all shrink-0"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="p-2 rounded-xl text-[#00ff41] hover:bg-[#00ff41]/10 transition-all disabled:opacity-20 disabled:cursor-not-allowed shrink-0"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                  </svg>
                </button>
              )}
            </div>

            <div className="text-[10px] text-[#777] mt-2 text-center">
              {statusText()}
            </div>
            {voiceInputError && conversationMode && (
              <div className="text-[10px] text-[#ff7777] mt-1 text-center">
                {voiceInputError}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
