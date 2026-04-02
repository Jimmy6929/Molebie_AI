"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getToken, logout as authLogout } from "@/lib/auth";
import {
  sendMessageStream,
  createSession,
  listSessions,
  getSessionMessages,
  deleteSession,
  renameSession,
  pinSession,
  uploadDocument,
  listDocuments,
  deleteDocument,
  attachDocumentToSession,
  listSessionAttachments,
  removeSessionAttachment,
  fileToDataUri,
  fetchImage,
  type SessionInfo,
  type ChatMessage,
  type SearchSource,
  type DocumentInfo,
  type SessionAttachmentInfo,
} from "@/lib/gateway";
import {
  useSpeechRecognition,
  useKokoroTTS,
  useVoiceSettings,
  isStopCommand,
} from "@/lib/voice";
import { useIsMobile } from "@/lib/useMediaQuery";
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
  sources?: SearchSource[];
  imageUrl?: string | null;
  imageId?: string | null;
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
// sttMode       — "Voice" button: mic → transcribe → fill input. No auto-send, no TTS.
// chatMode      — "Chat" button: mic → transcribe → auto-send → TTS → repeat.
//                 Auto-sends transcribed speech. Say "stop" to end.
// They are mutually exclusive.

export default function ChatPage() {
  const router = useRouter();
  const isMobile = useIsMobile();
  const [token, setToken] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string>("");
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try { return localStorage.getItem("molebie_web_search") === "true"; }
    catch { return false; }
  });
  const [mode, setMode] = useState<"instant" | "thinking" | "thinking_harder">("thinking");
  const [toolsOpen, setToolsOpen] = useState(false);

  // ── Mode flags ────────────────────────────────────────────────────────────
  const [sttMode, setSttMode] = useState(false);       // Voice button
  const [chatMode, setChatMode] = useState(false); // Chat button

  // ── Chat sub-settings ───────────────────────────────────────────────────
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Open sidebar by default on desktop after hydration
  useEffect(() => { if (!isMobile) setSidebarOpen(true); }, [isMobile]);
  // Persist web search toggle to localStorage
  useEffect(() => {
    try { localStorage.setItem("molebie_web_search", String(webSearchEnabled)); }
    catch {}
  }, [webSearchEnabled]);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [showImageToast, setShowImageToast] = useState(false);

  // ── Documents / RAG ──────────────────────────────────────────────────────
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docPanelOpen, setDocPanelOpen] = useState(false);
  const [docUploading, setDocUploading] = useState(false);
  const [docUploadProgress, setDocUploadProgress] = useState<number | null>(null);
  const [docUploadFilename, setDocUploadFilename] = useState("");
  const [docToast, setDocToast] = useState<string | null>(null);

  // ── Session Attachments ("Attach to Chat") ────────────────────────────
  const [attachments, setAttachments] = useState<SessionAttachmentInfo[]>([]);
  const [attachUploading, setAttachUploading] = useState(false);
  const [attachUploadProgress, setAttachUploadProgress] = useState<number | null>(null);
  const [attachUploadFilename, setAttachUploadFilename] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docFileInputRef = useRef<HTMLInputElement>(null);
  const attachFileInputRef = useRef<HTMLInputElement>(null);
  const isStreamingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const userScrolledUpRef = useRef(false);
  const sendVoiceMessageRef = useRef<(text: string) => void>(() => {});

  // chatLoopActive: should the Chat loop keep restarting the mic?
  const chatLoopRef = useRef(false);

  const { settings: voiceSettings, setSettings: setVoiceSettings } = useVoiceSettings();
  const { isSpeaking, createStreamingSpeaker, cancel: kokoroCancel } = useKokoroTTS();

  // ── Transcript handler ───────────────────────────────────────────────────
  const onFinalTranscript = useCallback(
    (text: string) => {
      const clean = text.trim();
      if (!clean) return;

      if (chatMode) {
        // Stop command — end Chat mode
        if (isStopCommand(clean)) {
          chatLoopRef.current = false;
          setChatMode(false);
          return;
        }

        // Chat mode — auto-send transcribed text
        setInput(clean);
        sendVoiceMessageRef.current(clean);
        return;
      }

      // STT mode — just fill the input box
      if (sttMode) {
        setInput(clean);
      }
    },
    [chatMode, sttMode]
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
    autoStopOnSilence: chatMode || sttMode,
  });

  // ── Chat auto-restart loop ──────────────────────────────────────────────
  // After each cycle (TTS finishes, transcription done, no active state)
  // auto-restart the mic so the conversation keeps going.
  useEffect(() => {
    if (!chatMode) return;
    if (!chatLoopRef.current) return;
    if (isListening || isTranscribing || isSpeaking || loading) return;

    const timer = setTimeout(() => {
      if (chatLoopRef.current && chatMode) {
        void startListening();
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [chatMode, isListening, isTranscribing, isSpeaking, loading, startListening]);

  // ── Auth ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const storedToken = getToken();
    if (!storedToken) {
      router.replace("/login");
      return;
    }
    setToken(storedToken);
    // Decode email from JWT payload (base64)
    try {
      const payload = JSON.parse(atob(storedToken.split(".")[1]));
      setUserEmail(payload.email || "");
    } catch {
      // token is malformed, redirect to login
      authLogout();
      router.replace("/login");
    }
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

  useEffect(() => { loadSessions(); }, [loadSessions]);

  const loadDocuments = useCallback(async () => {
    if (!token) return;
    try {
      const data = await listDocuments(token);
      setDocuments(data.documents);
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }, [token]);

  useEffect(() => { loadDocuments(); }, [loadDocuments]);

  async function handleDocUpload(file: File) {
    if (!token) return;
    setDocUploading(true);
    setDocUploadProgress(0);
    setDocUploadFilename(file.name);
    setDocToast(null);
    try {
      const result = await uploadDocument(token, file, (pct) => {
        setDocUploadProgress(pct);
        // Once upload reaches 100%, switch to indeterminate (server processing)
        if (pct >= 100) setDocUploadProgress(null);
      });
      setDocToast(`${result.filename} — ${result.chunks} chunks created`);
    } catch (err) {
      setDocToast(`Upload failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setDocUploading(false);
      setDocUploadProgress(null);
      setDocUploadFilename("");
      loadDocuments();
      setTimeout(() => setDocToast(null), 5000);
    }
  }

  async function handleDocDelete(docId: string) {
    if (!token) return;
    try {
      await deleteDocument(token, docId);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
    } catch (err) {
      console.error("Delete document error:", err);
    }
  }

  // ── Session Attachment handlers ────────────────────────────────────────

  const loadAttachments = useCallback(async (sessionId: string) => {
    if (!token || !sessionId) { setAttachments([]); return; }
    try {
      const data = await listSessionAttachments(token, sessionId);
      setAttachments(data.attachments);
    } catch (err) {
      console.error("Failed to load attachments:", err);
      setAttachments([]);
    }
  }, [token]);

  useEffect(() => {
    if (activeSessionId) loadAttachments(activeSessionId);
    else setAttachments([]);
  }, [activeSessionId, loadAttachments]);

  async function handleAttach(file: File) {
    if (!token) return;

    // Auto-create session if none exists (ChatGPT-like UX)
    let sessionId = activeSessionId;
    if (!sessionId) {
      try {
        const session = await createSession(token);
        sessionId = session.id;
        setActiveSessionId(session.id);
        loadSessions();
      } catch (err) {
        setDocToast("Failed to create session");
        setTimeout(() => setDocToast(null), 4000);
        return;
      }
    }

    setAttachUploading(true);
    setAttachUploadProgress(0);
    setAttachUploadFilename(file.name);
    setDocToast(null);
    try {
      const result = await attachDocumentToSession(token, sessionId, file, (pct) => {
        setAttachUploadProgress(pct);
        if (pct >= 100) setAttachUploadProgress(null);
      });
      setDocToast(result.truncated ? result.message : `${result.filename} attached`);
    } catch (err) {
      setDocToast(`Attach failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setAttachUploading(false);
      setAttachUploadProgress(null);
      setAttachUploadFilename("");
      loadAttachments(sessionId);
      setTimeout(() => setDocToast(null), 5000);
    }
  }

  async function handleRemoveAttachment(attachmentId: string) {
    if (!token || !activeSessionId) return;
    try {
      await removeSessionAttachment(token, activeSessionId, attachmentId);
      setAttachments((prev) => prev.filter((a) => a.id !== attachmentId));
    } catch (err) {
      console.error("Remove attachment error:", err);
    }
  }

  useEffect(() => {
    if (!token || !activeSessionId) { setMessages([]); return; }
    if (isStreamingRef.current) return;
    async function loadMessages() {
      try {
        const msgs = await getSessionMessages(token!, activeSessionId!);
        const displayMsgs: DisplayMessage[] = msgs.map((m: ChatMessage) => {
          let content = m.content;
          if (m.reasoning_content) content = `<think>${m.reasoning_content}</think>${content}`;
          return {
            id: m.id,
            role: m.role as DisplayMessage["role"],
            content,
            mode_used: m.mode_used,
            model_used: m.model_used ?? null,
            imageId: m.image_id ?? null,
            sources: m.sources ?? undefined,
          };
        });
        setMessages(displayMsgs);

        // Load images for messages that have them
        for (const msg of displayMsgs) {
          if (msg.imageId) {
            fetchImage(token!, msg.imageId).then((blobUrl) => {
              setMessages((prev) =>
                prev.map((m) => m.id === msg.id ? { ...m, imageUrl: blobUrl } : m)
              );
            }).catch((err) => console.error("Failed to load image:", err));
          }
        }
      } catch (err) { console.error("Failed to load messages:", err); }
    }
    loadMessages();
  }, [token, activeSessionId]);

  // ── Scroll detection ─────────────────────────────────────────────────────
  // Listen to wheel/touch (user-initiated only). Use requestAnimationFrame
  // so we read scroll position AFTER the browser has applied the scroll.
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    function checkPosition() {
      requestAnimationFrame(() => {
        const el = container!;
        const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
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

  // ── Auto-scroll on new content ──────────────────────────────────────────
  // overflow-anchor:none (via CSS) prevents browser from pushing viewport
  // when content grows. We only scroll-to-bottom when user is at bottom.
  useEffect(() => {
    if (userScrolledUpRef.current) return;
    const el = messagesContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

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
    const maxH = isMobile ? 120 : 200;
    if (el.scrollHeight > maxH) { el.style.height = maxH + "px"; el.style.overflowY = "auto"; }
    else { el.style.height = el.scrollHeight + "px"; el.style.overflowY = "hidden"; }
  }, [isMobile]);

  useEffect(() => { resizeTextarea(); }, [input, resizeTextarea]);

  // ── Core send function ───────────────────────────────────────────────────
  async function sendMessageWithText(rawInput: string) {
    if (!rawInput.trim() || !token || loading) return;

    const userMessage = rawInput.trim();

    // Convert attached image to base64 data URI
    let imageDataUri: string | undefined;
    let localImagePreview: string | null = null;
    if (imageFile) {
      imageDataUri = await fileToDataUri(imageFile);
      localImagePreview = imagePreview;
      removeImage();
    }

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

    setMessages((prev) => [
      ...prev,
      { id: `temp-${now}`, role: "user", content: userMessage, imageUrl: localImagePreview },
      {
        id: `stream-${now}`,
        role: "assistant",
        content: "",
        streaming: true,
        mode_used: chatMode ? "instant" : mode,
        streamStartedAt: now,
      },
    ]);
    isStreamingRef.current = true;

    try {
      if (chatMode) kokoroCancel();

      // Start streaming TTS — sentences play as soon as they arrive from the LLM
      const speaker = chatMode && token
        ? createStreamingSpeaker(token, voiceSettings.voiceId, voiceSettings.speed)
        : null;

      await sendMessageStream(
        token,
        userMessage,
        chatMode ? "instant" : mode,
        activeSessionId || undefined,
        chatMode,
        (content) => {
          setMessages((prev) => prev.map((m) => m.streaming ? { ...m, content } : m));
          speaker?.feed(content);
        },
        (sid) => { setActiveSessionId(sid); },
        controller.signal,
        () => setIsSearching(true),
        (sources) => {
          setIsSearching(false);
          setMessages((prev) =>
            prev.map((m) => m.streaming ? { ...m, sources } : m)
          );
        },
        imageDataUri,
        webSearchEnabled,
      );
      setMessages((prev) => prev.map((m) => m.streaming ? { ...m, streaming: false } : m));

      // Flush final partial sentence and wait for all playback to finish
      if (speaker) {
        speaker.finish();
        await speaker.done;
      }

      loadSessions();
    } catch (err) {
      if (controller.signal.aborted) {
        setMessages((prev) => prev.map((m) => m.streaming ? { ...m, streaming: false } : m));
      } else {
        console.error("Send error:", err);
        setMessages((prev) =>
          prev.map((m) =>
            m.streaming
              ? { ...m, content: `error: ${err instanceof Error ? err.message : "connection failed"}`, streaming: false }
              : m
          )
        );
      }
    } finally {
      abortControllerRef.current = null;
      isStreamingRef.current = false;
      setIsSearching(false);
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  sendVoiceMessageRef.current = (text: string) => { void sendMessageWithText(text); };

  function handleSend() { void sendMessageWithText(input); }

  function handleStop() {
    abortControllerRef.current?.abort();
    kokoroCancel();
    chatLoopRef.current = false;
  }

  // ── Regenerate last response ───────────────────────────────────────────
  async function handleRegenerate() {
    if (loading || !token || !activeSessionId) return;
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUserMsg) return;

    const lastAssistantMsg = [...messages].reverse().find((m) => m.role === "assistant");
    const regenerateMode = (lastAssistantMsg?.mode_used as "instant" | "thinking" | "thinking_harder") || mode;

    // Remove last assistant message from display
    setMessages((prev) => {
      const idx = prev.findLastIndex((m) => m.role === "assistant");
      return idx === -1 ? prev : prev.slice(0, idx);
    });

    setLoading(true);
    userScrolledUpRef.current = false;
    setShowScrollBtn(false);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const now = Date.now();

    setMessages((prev) => [
      ...prev,
      {
        id: `regen-${now}`,
        role: "assistant",
        content: "",
        streaming: true,
        mode_used: regenerateMode,
        streamStartedAt: now,
      },
    ]);
    isStreamingRef.current = true;

    try {
      await sendMessageStream(
        token,
        lastUserMsg.content,
        regenerateMode,
        activeSessionId,
        false,
        (content) => {
          setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, content } : m)));
        },
        () => {},
        controller.signal,
        () => setIsSearching(true),
        (sources) => {
          setIsSearching(false);
          setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, sources } : m)));
        },
        undefined,
        webSearchEnabled,
      );
      setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
      loadSessions();
    } catch (err) {
      if (controller.signal.aborted) {
        setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.streaming
              ? { ...m, content: `error: ${err instanceof Error ? err.message : "connection failed"}`, streaming: false }
              : m,
          ),
        );
      }
    } finally {
      abortControllerRef.current = null;
      isStreamingRef.current = false;
      setIsSearching(false);
      setLoading(false);
    }
  }

  // ── Export conversation as Markdown ─────────────────────────────────────
  function handleExportMarkdown() {
    if (messages.length === 0) return;
    const sessionTitle = sessions.find((s) => s.id === activeSessionId)?.title || "conversation";
    const lines = [`# ${sessionTitle}\n`];
    for (const msg of messages) {
      const label = msg.role === "user" ? "**You**" : "**AI**";
      lines.push(`${label}:\n${msg.content}\n\n---\n`);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sessionTitle.replace(/[^a-zA-Z0-9]/g, "_")}-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ── Mode toggles ─────────────────────────────────────────────────────────

  function enableSttMode() {
    // Disable Chat mode if active
    if (chatMode) {
      chatLoopRef.current = false;
      kokoroCancel();
      abortListening();
      setChatMode(false);
      setVoiceSettingsOpen(false);
    }
    setSttMode(true);
  }

  function disableSttMode() {
    abortListening();
    setSttMode(false);
  }

  function enableChatMode() {
    // Disable STT if active
    if (sttMode) {
      abortListening();
      setSttMode(false);
    }
    chatLoopRef.current = true;
    setChatMode(true);
    // The auto-restart useEffect will start the mic in 300ms
  }

  function disableChatMode() {
    chatLoopRef.current = false;
    setChatMode(false);
    kokoroCancel();
    abortListening();
    setVoiceSettingsOpen(false);
  }

  // Cleanup when modes are turned off
  useEffect(() => {
    if (!chatMode) {
      chatLoopRef.current = false;
    }
  }, [chatMode]);

  useEffect(() => {
    if (!sttMode && !chatMode) {
      abortListening();
    }
  }, [sttMode, chatMode, abortListening]);

  // ── STT mic button ────────────────────────────────────────────────────────
  function handleSttMicClick() {
    if (!supportsSpeechRecognition || loading || isTranscribing) return;
    if (isListening) { stopListening(); return; }
    void startListening();
  }

  // ── Chat mic button (manual override) ───────────────────────────────────
  function handleChatMicClick() {
    if (!supportsSpeechRecognition || loading || isTranscribing || isSpeaking) return;
    if (isListening) { stopListening(); return; }
    kokoroCancel();
    void startListening();
  }

  // ── Image handling ───────────────────────────────────────────────────────
  const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"];

  function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || !ALLOWED_IMAGE_TYPES.includes(file.type)) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }

  function removeImage() {
    setImageFile(null);
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handlePaste(e: React.ClipboardEvent) {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of Array.from(items)) {
      if (item.type.startsWith("image/") && ALLOWED_IMAGE_TYPES.includes(item.type)) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) {
          setImageFile(file);
          setImagePreview(URL.createObjectURL(file));
        }
        break;
      }
    }
  }

  const [isDragging, setIsDragging] = useState(false);

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.types.includes("Files")) setIsDragging(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && ALLOWED_IMAGE_TYPES.includes(file.type)) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  }

  function handleLogout() {
    authLogout();
    router.replace("/login");
  }

  function handleNewChat() {
    setActiveSessionId(null);
    setMessages([]);
    if (isMobile) setSidebarOpen(false);
    inputRef.current?.focus();
  }

  async function handleDeleteSession(id: string) {
    if (!token) return;
    try {
      await deleteSession(token, id);
      if (activeSessionId === id) { setActiveSessionId(null); setMessages([]); }
      loadSessions();
    } catch (err) { console.error("Delete error:", err); }
  }

  async function handleRenameSession(id: string, title: string) {
    if (!token) return;
    try { await renameSession(token, id, title); loadSessions(); }
    catch (err) { console.error("Rename error:", err); }
  }

  async function handlePinSession(id: string, pinned: boolean) {
    if (!token) return;
    try { await pinSession(token, id, pinned); loadSessions(); }
    catch (err) { console.error("Pin error:", err); }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  // ── Status hint text ─────────────────────────────────────────────────────
  function statusText(): string {
    if (chatMode) {
      if (isSpeaking) return "Chat is speaking...";
      if (isTranscribing) return "Transcribing...";
      if (isListening) return "Listening...";
      if (isSearching) return "Searching the web...";
      if (loading) return "Thinking...";
      return 'Listening loop active · say "stop" or "goodbye" to end';
    }
    if (sttMode) {
      if (isTranscribing) return "Transcribing...";
      if (isListening) return "Recording... auto-stops on silence";
      return "Voice mode · tap mic to record · Enter to send";
    }
    return "Enter to send · Shift+Enter for new line";
  }

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glow text-sm">authenticating...</div>
      </div>
    );
  }

  const chatActive = chatMode && (isListening || isTranscribing || isSpeaking || loading);

  return (
    <div className="min-h-screen flex bg-[#0a0a0a]">
      {/* Mobile: overlay drawer with backdrop */}
      {isMobile && sidebarOpen && (
        <div className="fixed inset-0 z-40 flex animate-fade-in">
          <div className="animate-slide-in-left">
            <Sidebar
              sessions={sessions}
              activeSessionId={activeSessionId}
              onSelectSession={(id) => { setActiveSessionId(id); setSidebarOpen(false); }}
              onNewChat={handleNewChat}
              onDeleteSession={handleDeleteSession}
              onRenameSession={handleRenameSession}
              onPinSession={handlePinSession}
              onLogout={handleLogout}
              userEmail={userEmail}
            />
          </div>
          <div className="flex-1 bg-black/50" onClick={() => setSidebarOpen(false)} />
        </div>
      )}
      {/* Desktop: inline sidebar with border */}
      {!isMobile && sidebarOpen && (
        <div className="border-r border-white/[0.06]">
          <Sidebar
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={setActiveSessionId}
            onNewChat={handleNewChat}
            onDeleteSession={handleDeleteSession}
            onRenameSession={handleRenameSession}
            onPinSession={handlePinSession}
            onLogout={handleLogout}
            userEmail={userEmail}
          />
        </div>
      )}

      <div className="flex-1 flex flex-col min-h-screen relative">
        {/* Header */}
        <header className="px-5 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-[#00ff41] hover:text-[#33ff66] transition-colors p-2 rounded-xl hover:bg-white/[0.06]"
              aria-label="Toggle sidebar"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <span className="text-xs text-[#999]">
              {activeSessionId ? `${activeSessionId.slice(0, 8)}...` : "New session"}
            </span>
          </div>
          {messages.length > 0 && (
            <button
              onClick={handleExportMarkdown}
              className="text-[#888] hover:text-[#ccc] transition-colors p-2 rounded-xl hover:bg-white/[0.06]"
              title="Export as Markdown"
              aria-label="Export as Markdown"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </button>
          )}
        </header>

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3 scroll-anchor-none">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-[#999] text-sm">What can I help you with?</div>
            </div>
          )}
          {messages.map((msg, idx) => {
            const isLastAssistant =
              msg.role === "assistant" &&
              !msg.streaming &&
              idx === messages.findLastIndex((m) => m.role === "assistant");
            return (
              <MessageBubble
                key={msg.id}
                role={msg.role}
                content={msg.content}
                streaming={msg.streaming}
                mode={msg.mode_used}
                model={msg.model_used}
                streamStartedAt={msg.streamStartedAt}
                sources={msg.sources}
                isSearching={msg.streaming && isSearching}
                imageUrl={msg.imageUrl}
                onRegenerate={isLastAssistant && !loading ? handleRegenerate : undefined}
              />
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        {showScrollBtn && (
          <div className="absolute bottom-28 md:bottom-32 left-1/2 -translate-x-1/2 z-10 animate-fade-in-up">
            <button onClick={scrollToBottom} className="glass rounded-full p-2.5 text-[#00ff41] hover:text-[#33ff66] transition-all hover:scale-105 shadow-lg shadow-black/40" aria-label="Scroll to bottom">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 pb-5 shrink-0 safe-bottom" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
          <div className="max-w-3xl mx-auto">

            {/* Chat Voice Settings panel */}
            <VoiceSettings
              open={voiceSettingsOpen}
              settings={voiceSettings}
              onChange={setVoiceSettings}
              onClose={() => setVoiceSettingsOpen(false)}
            />

            {/* Brain panel */}
            {docPanelOpen && (
              <div className="glass rounded-2xl p-3 mb-2 animate-fade-in max-h-48 overflow-y-auto">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] text-[#aaa] font-medium tracking-wide">Brain</span>
                  <button
                    onClick={() => docFileInputRef.current?.click()}
                    disabled={docUploading}
                    className="text-[10px] px-2 py-1 rounded-lg bg-[#33ccff]/15 text-[#66ddff] hover:bg-[#33ccff]/25 transition-all disabled:opacity-40"
                  >
                    {docUploading ? "Processing..." : "+ Upload"}
                  </button>
                </div>
                {docUploading && (
                  <div className="mb-2 animate-fade-in">
                    <div className="w-full h-1 rounded-full overflow-hidden bg-white/[0.06]">
                      {docUploadProgress !== null && docUploadProgress < 100 ? (
                        <div
                          className="h-full bg-[#00ff41] rounded-full transition-all duration-300"
                          style={{ width: `${docUploadProgress}%` }}
                        />
                      ) : (
                        <div className="h-full w-1/4 bg-[#00ff41] rounded-full animate-indeterminate" />
                      )}
                    </div>
                    <p className="text-[10px] text-[#888] mt-1">
                      {docUploadProgress !== null && docUploadProgress < 100
                        ? `Uploading ${docUploadFilename}... ${docUploadProgress}%`
                        : `Processing ${docUploadFilename}...`}
                    </p>
                  </div>
                )}
                {documents.length === 0 && !docUploading ? (
                  <p className="text-[11px] text-[#777]">No documents uploaded. Upload TXT, MD, PDF, or DOCX files to give Chat long-term memory.</p>
                ) : (
                  <div className="space-y-1.5">
                    {documents.map((doc) => (
                      <div key={doc.id} className="flex items-center justify-between gap-2 group">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                            doc.status === "completed" ? "bg-[#00ff41]" : doc.status === "processing" ? "bg-[#ffcc33] animate-pulse" : doc.status === "failed" ? "bg-[#ff4444]" : "bg-[#888]"
                          }`} />
                          <span className="text-[11px] text-[#ccc] truncate">{doc.filename}</span>
                          <span className="text-[9px] text-[#666] shrink-0">{(doc.file_size / 1024).toFixed(0)}KB</span>
                        </div>
                        <button
                          onClick={() => handleDocDelete(doc.id)}
                          className="text-[#666] hover:text-[#ff4444] md:opacity-0 md:group-hover:opacity-100 transition-all shrink-0"
                          title="Delete document"
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Attachment progress bar */}
            {attachUploading && (
              <div className="mb-2 glass rounded-xl px-3 py-2 animate-fade-in">
                <div className="w-full h-1 rounded-full overflow-hidden bg-white/[0.06]">
                  {attachUploadProgress !== null && attachUploadProgress < 100 ? (
                    <div
                      className="h-full bg-[#00ff41] rounded-full transition-all duration-300"
                      style={{ width: `${attachUploadProgress}%` }}
                    />
                  ) : (
                    <div className="h-full w-1/4 bg-[#00ff41] rounded-full animate-indeterminate" />
                  )}
                </div>
                <p className="text-[10px] text-[#888] mt-1">
                  {attachUploadProgress !== null && attachUploadProgress < 100
                    ? `Uploading ${attachUploadFilename}... ${attachUploadProgress}%`
                    : `Attaching ${attachUploadFilename}...`}
                </p>
              </div>
            )}

            {/* Document upload toast */}
            {docToast && (
              <div className="mb-2 glass rounded-xl px-4 py-2 text-[11px] text-[#66ddff] animate-fade-in">
                {docToast}
              </div>
            )}

            {imagePreview && (
              <div className="mb-2 flex items-center gap-2 animate-fade-in">
                <div className="relative group">
                  <img src={imagePreview} alt="Selected" className="h-16 w-16 object-cover rounded-xl border border-white/[0.06]" />
                  <button onClick={removeImage} className="absolute -top-1.5 -right-1.5 bg-[#ff3333] text-white rounded-full w-5 h-5 flex items-center justify-center text-[10px] md:opacity-0 md:group-hover:opacity-100 transition-opacity" aria-label="Remove image">×</button>
                </div>
                <span className="text-[10px] text-[#999]">{imageFile?.name}</span>
              </div>
            )}

            {/* Attached files chips */}
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-2 animate-fade-in">
                {attachments.map((att) => (
                  <div
                    key={att.id}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#00ff41]/10 border border-[#00ff41]/20 text-[11px] text-[#00ff41] group"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                    </svg>
                    <span className="truncate max-w-[140px]">{att.filename}</span>
                    <span className="text-[9px] text-[#00ff41]/60">{(att.content_length / 1000).toFixed(1)}k</span>
                    <button
                      onClick={() => handleRemoveAttachment(att.id)}
                      className="text-[#00ff41]/40 hover:text-[#ff4444] transition-colors ml-0.5"
                      title="Remove attachment"
                    >
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}

            {isDragging && (
              <div className="mb-2 glass rounded-2xl p-6 border-2 border-dashed border-[#00ff41]/50 flex items-center justify-center animate-fade-in">
                <span className="text-sm text-[#00ff41]/70">Drop image here</span>
              </div>
            )}

            <div className={`glass rounded-2xl p-2 flex flex-col gap-2 transition-all ${isDragging ? "border border-[#00ff41]/30" : ""}`}>
              {/* Tool buttons */}
              <div className="flex items-center gap-1.5 flex-wrap">
                {/* Hidden file inputs */}
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />
                <input
                  ref={attachFileInputRef}
                  type="file"
                  accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleAttach(f);
                    e.target.value = "";
                  }}
                  className="hidden"
                />
                <input
                  ref={docFileInputRef}
                  type="file"
                  accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleDocUpload(f);
                    e.target.value = "";
                  }}
                  className="hidden"
                />

                {/* ── EXPANDABLE TOOLS MENU ──────────────────────────────── */}
                <div className="flex items-center gap-1.5">
                  {/* "+" toggle button */}
                  <button
                    onClick={() => setToolsOpen(prev => !prev)}
                    className={`p-2 rounded-xl transition-all duration-200 shrink-0 ${
                      toolsOpen
                        ? "bg-white/[0.08] text-[#f0f0f0] rotate-45"
                        : "text-[#999] hover:text-[#f0f0f0] hover:bg-white/[0.06]"
                    }`}
                    title={toolsOpen ? "Close tools" : "Open tools"}
                    aria-label={toolsOpen ? "Close tools" : "Open tools"}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  </button>

                  {/* Expandable tool icons */}
                  <div className={`flex items-center gap-1.5 overflow-hidden transition-all duration-300 ease-out ${
                    toolsOpen ? "max-w-[300px] opacity-100" : "max-w-0 opacity-0"
                  }`}>
                    {/* Image upload */}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={mode === "instant"}
                      className={`p-2 rounded-xl transition-all shrink-0 ${
                        mode === "instant"
                          ? "text-[#555] cursor-not-allowed opacity-40"
                          : "text-[#999] hover:text-[#00ff41] hover:bg-white/[0.06]"
                      }`}
                      title={mode === "instant" ? "Image not supported in Fast mode" : "Add image"}
                      aria-label={mode === "instant" ? "Image not supported in Fast mode" : "Add image"}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="4" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                      </svg>
                    </button>

                    {/* Attach file to chat */}
                    <button
                      onClick={() => attachFileInputRef.current?.click()}
                      disabled={attachUploading}
                      className={`p-2 rounded-xl transition-all shrink-0 ${
                        attachments.length > 0
                          ? "text-[#00ff41] hover:bg-[#00ff41]/10"
                          : "text-[#999] hover:text-[#00ff41] hover:bg-white/[0.06]"
                      } disabled:opacity-40`}
                      title={attachUploading ? "Attaching..." : "Attach file to chat"}
                      aria-label={attachUploading ? "Attaching..." : "Attach file to chat"}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                      </svg>
                    </button>

                    {/* Brain / Documents */}
                    <button
                      onClick={() => setDocPanelOpen((p) => !p)}
                      className={`relative p-2 rounded-xl transition-all shrink-0 ${
                        docPanelOpen
                          ? "bg-[#33ccff]/20 text-[#66ddff]"
                          : documents.length > 0
                            ? "text-[#66ddff] hover:bg-[#33ccff]/10"
                            : "text-[#999] hover:text-[#66ddff] hover:bg-white/[0.06]"
                      }`}
                      title={`Brain${documents.length ? ` (${documents.length})` : ""}`}
                      aria-label={`Brain${documents.length ? ` (${documents.length} documents)` : ""}`}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M9.5 2a3.5 3.5 0 0 0-3.2 4.8A3.5 3.5 0 0 0 4 10.5a3.5 3.5 0 0 0 1.3 2.7A3.5 3.5 0 0 0 5 15a3.5 3.5 0 0 0 3.5 3.5h1V22h5v-3.5h1A3.5 3.5 0 0 0 19 15a3.5 3.5 0 0 0-.3-1.8A3.5 3.5 0 0 0 20 10.5a3.5 3.5 0 0 0-2.3-3.2A3.5 3.5 0 0 0 14.5 2h-5z" />
                        <path d="M12 2v20" />
                      </svg>
                      {documents.length > 0 && (
                        <span className="absolute -top-1 -right-1 bg-[#33ccff] text-black text-[8px] font-bold rounded-full w-3.5 h-3.5 flex items-center justify-center">{documents.length}</span>
                      )}
                    </button>

                    {/* Web Search */}
                    <button
                      onClick={() => setWebSearchEnabled(prev => !prev)}
                      className={`p-2 rounded-xl transition-all shrink-0 ${
                        webSearchEnabled
                          ? "bg-[#3399ff]/15 text-[#66bbff]"
                          : "text-[#999] hover:text-[#66bbff] hover:bg-white/[0.06]"
                      }`}
                      title={webSearchEnabled ? "Web search on" : "Web search off"}
                      aria-label="Toggle web search"
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="2" y1="12" x2="22" y2="12" />
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                      </svg>
                    </button>

                    {/* Chat conversation mode */}
                    <button
                      onClick={() => chatMode ? disableChatMode() : enableChatMode()}
                      className={`p-2 rounded-xl transition-all shrink-0 ${
                        chatMode
                          ? "bg-[#33ccff]/20 text-[#66ddff]"
                          : "text-[#999] hover:text-[#66ddff] hover:bg-white/[0.06]"
                      }`}
                      title="Chat conversation mode — voice chat with TTS"
                      aria-label="Chat conversation mode"
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* ── THINK MODE TOGGLE ──────────────────────────────────── */}
                <button
                  onClick={() => setMode(mode === "instant" ? "thinking" : "instant")}
                  className={`text-[11px] px-3 py-1.5 rounded-xl transition-all shrink-0 ${
                    mode !== "instant"
                      ? "bg-[#ffbb33]/15 text-[#ffcc33] border border-[#ffbb33]/30"
                      : "text-[#999] hover:text-[#00ff41] border border-white/[0.08] hover:border-white/[0.15]"
                  }`}
                >
                  {mode === "instant" ? "Fast" : mode === "thinking_harder" ? "Think+" : "Think"}
                </button>
                {mode !== "instant" && (
                  <button
                    onClick={() => setMode(mode === "thinking_harder" ? "thinking" : "thinking_harder")}
                    className={`text-[11px] px-3 py-1.5 rounded-xl transition-all shrink-0 ${
                      mode === "thinking_harder"
                        ? "bg-[#ff8833]/20 text-[#ff9955] border border-[#ff8833]/40"
                        : "text-[#ffcc33] border border-[#ffbb33]/30 hover:bg-[#ffbb33]/10"
                    }`}
                  >
                    {mode === "thinking_harder" ? "Think Harder" : "Normal"}
                  </button>
                )}

                {/* ── VOICE BUTTON (STT only) ─────────────────────────────── */}
                <button
                  onClick={() => sttMode ? disableSttMode() : enableSttMode()}
                  className={`p-2 rounded-xl transition-all shrink-0 ${
                    sttMode
                      ? "bg-[#00ff41]/20 text-[#00ff41]"
                      : "text-[#999] hover:text-[#00ff41] hover:bg-white/[0.06]"
                  }`}
                  title="Voice input — speech to text"
                  aria-label="Voice input"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                  </svg>
                </button>

                {/* STT mic button — only visible in stt mode */}
                {sttMode && (
                  <button
                    onClick={handleSttMicClick}
                    disabled={!supportsSpeechRecognition || loading || isTranscribing}
                    className={`p-2 rounded-xl transition-all shrink-0 ${
                      isListening
                        ? "bg-[#ff4444]/20 text-[#ff6666] animate-pulse"
                        : isTranscribing
                          ? "bg-[#ffbb33]/15 text-[#ffcc33] animate-pulse"
                          : "text-[#00ff41] hover:bg-[#00ff41]/10"
                    } disabled:opacity-30 disabled:cursor-not-allowed`}
                    title={isTranscribing ? "Transcribing..." : isListening ? "Stop recording" : "Record voice"}
                    aria-label={isTranscribing ? "Transcribing..." : isListening ? "Stop recording" : "Record voice"}
                  >
                    {isTranscribing ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" />
                      </svg>
                    ) : (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                        <line x1="12" y1="19" x2="12" y2="23" />
                        <line x1="8" y1="23" x2="16" y2="23" />
                      </svg>
                    )}
                  </button>
                )}

                {/* Chat controls — only visible in Chat mode */}
                {chatMode && (
                  <>
                    {/* Manual mic button */}
                    <button
                      onClick={handleChatMicClick}
                      disabled={!supportsSpeechRecognition || loading || isTranscribing || isSpeaking}
                      className={`p-2 rounded-xl transition-all shrink-0 ${
                        isListening
                          ? "bg-[#ff4444]/20 text-[#ff6666] animate-pulse"
                          : isTranscribing
                            ? "bg-[#ffbb33]/15 text-[#ffcc33] animate-pulse"
                            : isSpeaking
                              ? "bg-[#33ccff]/15 text-[#66ddff] animate-pulse"
                              : "text-[#66ddff] hover:bg-[#33ccff]/10"
                      } disabled:opacity-30 disabled:cursor-not-allowed`}
                      title={isSpeaking ? "Chat is speaking" : isTranscribing ? "Transcribing..." : isListening ? "Stop" : "Speak"}
                      aria-label={isSpeaking ? "Chat is speaking" : isTranscribing ? "Transcribing..." : isListening ? "Stop" : "Speak"}
                    >
                      {isSpeaking ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                          <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                          <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                        </svg>
                      ) : isTranscribing ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" />
                        </svg>
                      ) : (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                          <line x1="12" y1="19" x2="12" y2="23" />
                          <line x1="8" y1="23" x2="16" y2="23" />
                        </svg>
                      )}
                    </button>

                    {/* Stop conversation button — only when something is happening */}
                    {chatActive && (
                      <button
                        onClick={disableChatMode}
                        className="p-2 rounded-xl bg-[#ff4444]/15 text-[#ff6666] hover:bg-[#ff4444]/25 transition-all shrink-0"
                        title="End conversation"
                        aria-label="End conversation"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                          <rect x="6" y="6" width="12" height="12" rx="2" />
                        </svg>
                      </button>
                    )}

                    {/* Voice settings gear */}
                    <button
                      onClick={() => setVoiceSettingsOpen((p) => !p)}
                      className={`p-2 rounded-xl transition-all shrink-0 ${
                        voiceSettingsOpen
                          ? "bg-[#33ccff]/20 text-[#66ddff]"
                          : "text-[#66ddff] hover:bg-[#33ccff]/10"
                      }`}
                      title="Chat settings"
                      aria-label="Chat settings"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="3" />
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                      </svg>
                    </button>
                  </>
                )}
              </div>

              {/* Text input + send — always full width row */}
              <div className="flex items-end gap-2 w-full">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => { setInput(e.target.value); resizeTextarea(); }}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  placeholder="Type anything..."
                  rows={1}
                  className="flex-1 bg-transparent text-[#f0f0f0] text-sm resize-none focus:outline-none placeholder-[#888] min-h-[36px] max-h-[200px] py-2 px-1"
                  style={{ overflowY: "hidden" }}
                  disabled={loading}
                  autoFocus
                />

                {/* Send / Stop button */}
                {loading ? (
                  <button onClick={handleStop} className="p-2 rounded-xl bg-[#ff5555]/15 text-[#ff5555] hover:bg-[#ff5555]/25 transition-all shrink-0" aria-label="Stop generating">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  </button>
                ) : (
                  <button onClick={handleSend} disabled={!input.trim() && !imageFile} className="p-2 rounded-xl text-[#00ff41] hover:bg-[#00ff41]/10 transition-all disabled:opacity-20 disabled:cursor-not-allowed shrink-0" aria-label="Send message">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="19" x2="12" y2="5" />
                      <polyline points="5 12 12 5 19 12" />
                    </svg>
                  </button>
                )}
              </div>
            </div>

            <div className="text-[10px] text-[#777] mt-2 text-center">{statusText()}</div>
            {voiceInputError && (sttMode || chatMode) && (
              <div className="text-[10px] text-[#ff7777] mt-1 text-center">{voiceInputError}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
