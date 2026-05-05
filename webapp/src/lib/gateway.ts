import { GATEWAY_URL } from "./gatewayUrl";

export interface ChatResponse {
  session_id: string;
  message: {
    id: string;
    role: string;
    content: string;
    mode_used: string | null;
    created_at: string;
  };
  session_title: string | null;
}

export interface SessionInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  is_archived: boolean;
  is_pinned: boolean;
}

export interface ChatMessage {
  id: string;
  role: string;
  content: string;
  mode_used: string | null;
  model_used?: string | null;
  reasoning_content?: string | null;
  image_id?: string | null;
  sources?: { title: string; url: string }[] | null;
  created_at: string;
}

async function apiCall<T>(
  path: string,
  token: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${GATEWAY_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  });

  if (!res.ok) {
    // Stale or invalid token — clear it and redirect to login
    if (res.status === 401) {
      if (typeof window !== "undefined") {
        localStorage.removeItem("molebie_token");
        window.location.href = "/login";
      }
    }
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}

export async function sendMessage(
  token: string,
  message: string,
  mode: "instant" | "thinking" | "thinking_harder" = "instant",
  sessionId?: string,
  conversationMode: boolean = false,
  image?: string,
  webSearch?: boolean,
): Promise<ChatResponse> {
  return apiCall<ChatResponse>("/chat", token, {
    method: "POST",
    body: JSON.stringify({
      message,
      mode,
      session_id: sessionId || null,
      conversation_mode: conversationMode,
      ...(image ? { image } : {}),
      web_search: webSearch ?? false,
    }),
  });
}

export interface SearchSource {
  title: string;
  url: string;
}

export async function sendMessageStream(
  token: string,
  message: string,
  mode: "instant" | "thinking" | "thinking_harder" = "instant",
  sessionId?: string,
  conversationMode: boolean = false,
  onChunk: (text: string) => void = () => {},
  onSessionId: (id: string) => void = () => {},
  signal?: AbortSignal,
  onSearchStart?: () => void,
  onSearchDone?: (sources: SearchSource[]) => void,
  image?: string,
  webSearch?: boolean,
): Promise<string> {
  const res = await fetch(`${GATEWAY_URL}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      message,
      mode,
      session_id: sessionId || null,
      conversation_mode: conversationMode,
      ...(image ? { image } : {}),
      web_search: webSearch ?? false,
    }),
    signal,
  });

  if (!res.ok) {
    if (res.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem("molebie_token");
      window.location.href = "/login";
    }
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  // Get session ID from header (if CORS exposes it)
  const sid = res.headers.get("X-Session-ID");
  if (sid) onSessionId(sid);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let sseBuffer = "";

  // Format A: server sends delta.reasoning_content (vLLM with --enable-reasoning,
  // some Ollama configs). Reasoning streams in a dedicated field.
  let reasoningField = "";
  let reasoningFieldUsed = false;

  // Format B/C: thinking arrives in delta.content. Format B has explicit
  // <think>...</think> tags; Format C has only the closing </think> because
  // the backend's chat template consumed the opening token.
  let rawContent = "";

  // Backend metadata: distinct from the user's UI mode, ``enable_thinking``
  // here reflects what the backend actually told the model to do — the
  // chat route can disable CoT (RAG auto-disable) even when the user
  // picked "thinking". Default true on thinking modes so the optimistic
  // Format-C wrap still fires if metadata never arrives (older backends).
  let backendEnableThinking = mode !== "instant";

  function processLine(line: string) {
    if (!line.startsWith("data: ") || line.includes("[DONE]")) return;

    let data;
    try {
      data = JSON.parse(line.slice(6));
    } catch {
      return;
    }

    if (data.session_id && !data.choices) {
      onSessionId(data.session_id);
      return;
    }

    if (data.metadata && typeof data.metadata.enable_thinking === "boolean") {
      backendEnableThinking = data.metadata.enable_thinking;
      return;
    }

    if (data.type === "search_start" && onSearchStart) {
      onSearchStart();
      return;
    }

    if (data.type === "search_done" && data.sources && onSearchDone) {
      onSearchDone(data.sources as SearchSource[]);
      return;
    }

    const delta = data.choices?.[0]?.delta;
    if (!delta) return;

    const rc = delta.reasoning_content;
    const ct = delta.content;

    if (rc) {
      reasoningField += rc;
      reasoningFieldUsed = true;
    }

    if (typeof ct === "string" && ct.length > 0) {
      rawContent += ct;
    }

    onChunk(synthesize());
  }

  function stripThinkingTags(text: string): string {
    const openIdx = text.indexOf("<think>");
    const closeIdx = text.indexOf("</think>");
    if (openIdx !== -1 && closeIdx !== -1) {
      return (text.slice(0, openIdx) + text.slice(closeIdx + "</think>".length)).trim();
    }
    if (closeIdx !== -1) {
      return text.slice(closeIdx + "</think>".length).trim();
    }
    if (openIdx !== -1) {
      return text.slice(0, openIdx).trim();
    }
    return text;
  }

  function synthesize(): string {
    if (mode === "instant") {
      let clean = rawContent;
      const closeIdx = clean.indexOf("</think>");
      if (closeIdx !== -1) {
        clean = clean.slice(closeIdx + "</think>".length);
      }
      clean = stripThinkingTags(clean);
      return clean.trimStart();
    }

    // Format A: reasoning streamed in delta.reasoning_content. Always emit
    // a closed <think> block so partial reasoning surfaces immediately.
    if (reasoningFieldUsed) {
      return `<think>${reasoningField}</think>${rawContent}`;
    }

    // Format B: explicit <think> in content — pass through; thinkParser
    // handles the unclosed-tag case for mid-stream rendering.
    if (rawContent.includes("<think>")) {
      return rawContent;
    }

    // Format C: opening <think> consumed by the chat template (mlx_vlm).
    // Once </think> appears the prefix was reasoning — re-inject the opener
    // so the parser routes thinking and response correctly.
    const closeIdx = rawContent.indexOf("</think>");
    if (closeIdx !== -1) {
      const thinkText = rawContent.slice(0, closeIdx).trim();
      const respText = rawContent.slice(closeIdx + "</think>".length);
      return (thinkText ? `<think>${thinkText}</think>` : "") + respText;
    }

    // Mid-stream, thinking mode active, no opener seen and no closer yet:
    // optimistically wrap so the panel populates in real time. Only fires
    // when the backend confirmed CoT is actually enabled — otherwise the
    // RAG auto-disable path (chat.py:1383) would falsely wrap a plain
    // response as thinking content.
    if (backendEnableThinking && rawContent.trim()) {
      return `<think>${rawContent}`;
    }

    return rawContent;
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    sseBuffer += decoder.decode(value, { stream: true });
    const parts = sseBuffer.split("\n");
    sseBuffer = parts.pop() || "";

    for (const part of parts) {
      const trimmed = part.trim();
      if (trimmed) processLine(trimmed);
    }
  }

  if (sseBuffer.trim()) processLine(sseBuffer.trim());

  // Final assembled result
  return synthesize();
}

export async function createSession(token: string): Promise<SessionInfo> {
  return apiCall<SessionInfo>("/chat/sessions/create", token, {
    method: "POST",
  });
}

export async function listSessions(
  token: string
): Promise<{ sessions: SessionInfo[] }> {
  return apiCall<{ sessions: SessionInfo[] }>("/chat/sessions", token);
}

export async function getSessionMessages(
  token: string,
  sessionId: string
): Promise<ChatMessage[]> {
  return apiCall<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`, token);
}

export async function renameSession(
  token: string,
  sessionId: string,
  title: string
): Promise<SessionInfo> {
  return apiCall<SessionInfo>(`/chat/sessions/${sessionId}`, token, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function pinSession(
  token: string,
  sessionId: string,
  isPinned: boolean
): Promise<SessionInfo> {
  return apiCall<SessionInfo>(`/chat/sessions/${sessionId}/pin`, token, {
    method: "PATCH",
    body: JSON.stringify({ is_pinned: isPinned }),
  });
}

export async function deleteSession(
  token: string,
  sessionId: string
): Promise<void> {
  await fetch(`${GATEWAY_URL}/chat/sessions/${sessionId}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

export async function fetchTTSAudio(
  token: string,
  text: string,
  voice: string = "bm_george",
  speed: number = 1.0
): Promise<Blob> {
  const res = await fetch(`${GATEWAY_URL}/chat/tts`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ text, voice, speed }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`TTS error ${res.status}: ${errText}`);
  }

  return res.blob();
}

export interface TranscribeResult {
  text: string;
  speaker_verified?: boolean;
  speaker_confidence?: number;
}

export async function transcribeAudio(
  token: string,
  audioBlob: Blob,
  verifySpeaker: boolean = false
): Promise<TranscribeResult> {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.webm");
  formData.append("verify_speaker", verifySpeaker ? "true" : "false");

  const res = await fetch(`${GATEWAY_URL}/chat/transcribe`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Transcription error ${res.status}: ${text}`);
  }

  return res.json();
}

export interface VoiceProfileStatus {
  enrolled: boolean;
  n_samples: number;
  complete: boolean;
  required: number;
}

export async function getVoiceProfileStatus(
  token: string
): Promise<VoiceProfileStatus> {
  return apiCall<VoiceProfileStatus>("/chat/voice-profile", token);
}

export async function enrollVoiceSample(
  token: string,
  audioBlob: Blob
): Promise<VoiceProfileStatus> {
  const formData = new FormData();
  formData.append("file", audioBlob, "enrollment.webm");

  const res = await fetch(`${GATEWAY_URL}/chat/voice-enroll`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Enrollment error ${res.status}: ${errText}`);
  }

  return res.json();
}

export async function deleteVoiceProfile(
  token: string
): Promise<{ deleted: boolean }> {
  return apiCall<{ deleted: boolean }>("/chat/voice-profile", token, {
    method: "DELETE",
  });
}

export async function checkInferenceHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(`${GATEWAY_URL}/health/inference`);
  return res.json();
}

// ── Documents / RAG ──────────────────────────────────────────────────────

export interface DocumentInfo {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  created_at: string;
  processed_at: string | null;
}

export interface UploadResponse {
  id: string;
  filename: string;
  status: string;
  chunks: number;
  message: string;
}

export async function uploadDocument(
  token: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload error ${xhr.status}: ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed — connection error"));

    xhr.open("POST", `${GATEWAY_URL}/documents/upload`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(formData);
  });
}

export async function listDocuments(
  token: string
): Promise<{ documents: DocumentInfo[] }> {
  return apiCall<{ documents: DocumentInfo[] }>("/documents", token);
}

export async function deleteDocument(
  token: string,
  documentId: string
): Promise<void> {
  await fetch(`${GATEWAY_URL}/documents/${documentId}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

// ── Session Attachments ("Attach to Chat") ──────────────────────────────

export interface SessionAttachmentInfo {
  id: string;
  session_id: string;
  filename: string;
  content_length: number;
  file_size: number;
  truncated: boolean;
  created_at: string;
}

export interface AttachResponse {
  id: string;
  filename: string;
  content_length: number;
  session_id: string;
  truncated: boolean;
  message: string;
}

export async function attachDocumentToSession(
  token: string,
  sessionId: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<AttachResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Attach error ${xhr.status}: ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Attach failed — connection error"));

    xhr.open("POST", `${GATEWAY_URL}/documents/sessions/${sessionId}/attach`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(formData);
  });
}

export async function listSessionAttachments(
  token: string,
  sessionId: string
): Promise<{ attachments: SessionAttachmentInfo[] }> {
  return apiCall<{ attachments: SessionAttachmentInfo[] }>(
    `/documents/sessions/${sessionId}/attachments`,
    token
  );
}

export async function removeSessionAttachment(
  token: string,
  sessionId: string,
  attachmentId: string
): Promise<void> {
  await fetch(
    `${GATEWAY_URL}/documents/sessions/${sessionId}/attachments/${attachmentId}`,
    {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    }
  );
}

// ── Image / Vision helpers ──────────────────────────────────────────────

const MAX_IMAGE_DIMENSION = 1024;

export function fileToDataUri(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;

      // Resize if either dimension exceeds the limit
      if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
        const scale = MAX_IMAGE_DIMENSION / Math.max(width, height);
        width = Math.round(width * scale);
        height = Math.round(height * scale);
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) { reject(new Error("Canvas context unavailable")); return; }
      ctx.drawImage(img, 0, 0, width, height);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

export function getImageUrl(token: string, imageId: string): string {
  return `${GATEWAY_URL}/chat/images/${imageId}?token=${encodeURIComponent(token)}`;
}

export async function fetchImage(token: string, imageId: string): Promise<string> {
  const res = await fetch(`${GATEWAY_URL}/chat/images/${imageId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Image fetch failed: ${res.status}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ── Folder Ingest ────────────────────────────────────────────────────────

export interface FolderManifestEntry {
  relative_path: string;
  size: number;
  content_type: string | null;
}

export interface FolderAcceptedFile {
  file_id: string;
  relative_path: string;
  size: number;
}

export interface FolderRejectedFile {
  relative_path: string;
  reason: string;
}

export interface StartFolderJobResponse {
  job_id: string;
  accepted_files: FolderAcceptedFile[];
  rejected_files: FolderRejectedFile[];
  total_accepted_bytes: number;
}

export interface UploadFolderBatchResponse {
  accepted: string[];
  rejected: FolderRejectedFile[];
}

export type FolderJobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface FolderJobSnapshot {
  job_id: string;
  status: FolderJobStatus;
  root_label: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  skipped_files: number;
  total_bytes: number;
  processed_bytes: number;
  started_at: string | null;
  finished_at: string | null;
  last_event_id: number;
}

export interface FolderEventCallbacks {
  onJobStarted?: (data: { job_id: string; total_files: number; total_bytes: number; root_label: string }) => void;
  onFileStarted?: (data: { file_id: string; relative_path: string; size: number }) => void;
  onFileCompleted?: (data: { file_id: string; relative_path: string; chunks: number; document_id: string; size: number }) => void;
  onFileFailed?: (data: { file_id: string; relative_path: string; error: string }) => void;
  onFileSkipped?: (data: { file_id: string; relative_path: string; reason: string }) => void;
  onProgress?: (data: {
    processed_files: number;
    failed_files: number;
    skipped_files: number;
    processed_bytes: number;
    total_files: number;
    total_bytes: number;
  }) => void;
  onJobCompleted?: (data: { processed_files: number; failed_files: number; skipped_files: number; total_bytes: number; duration_ms: number }) => void;
  onJobCancelled?: (data: { job_id: string; processed_files: number; failed_files: number }) => void;
  onJobFailed?: (data: { error: string }) => void;
  onError?: (err: Event) => void;
}

export async function startFolderUpload(
  token: string,
  rootLabel: string,
  manifest: FolderManifestEntry[],
): Promise<StartFolderJobResponse> {
  return apiCall<StartFolderJobResponse>("/documents/folder/start", token, {
    method: "POST",
    body: JSON.stringify({ root_label: rootLabel, files: manifest }),
  });
}

export async function uploadFolderBatch(
  token: string,
  jobId: string,
  files: { file: File; relativePath: string }[],
  onBatchProgress?: (loaded: number, total: number) => void,
): Promise<UploadFolderBatchResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    for (const { file, relativePath } of files) {
      formData.append("files", file, relativePath.split("/").pop() || file.name);
      formData.append("relative_paths", relativePath);
    }

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onBatchProgress) {
        onBatchProgress(e.loaded, e.total);
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (err) {
          reject(new Error(`Bad JSON: ${err}`));
        }
      } else {
        reject(new Error(`Upload error ${xhr.status}: ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed — connection error"));

    xhr.open("POST", `${GATEWAY_URL}/documents/folder/${encodeURIComponent(jobId)}/upload`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(formData);
  });
}

export interface FolderEventStreamHandle {
  close: () => void;
}

export function streamFolderEvents(
  token: string,
  jobId: string,
  callbacks: FolderEventCallbacks,
  lastEventId?: number,
): FolderEventStreamHandle {
  const url = new URL(`${GATEWAY_URL}/documents/folder/${encodeURIComponent(jobId)}/events`);
  url.searchParams.set("token", token);
  if (typeof lastEventId === "number" && lastEventId > 0) {
    url.searchParams.set("last_event_id", String(lastEventId));
  }
  const es = new EventSource(url.toString());

  const bind = <T,>(name: string, handler?: (d: T) => void) => {
    es.addEventListener(name, (ev) => {
      if (!handler) return;
      try {
        handler(JSON.parse((ev as MessageEvent).data) as T);
      } catch {
        /* malformed event — ignore */
      }
    });
  };

  bind("job_started", callbacks.onJobStarted);
  bind("file_started", callbacks.onFileStarted);
  bind("file_completed", callbacks.onFileCompleted);
  bind("file_failed", callbacks.onFileFailed);
  bind("file_skipped", callbacks.onFileSkipped);
  bind("progress", callbacks.onProgress);

  es.addEventListener("job_completed", (ev) => {
    try { callbacks.onJobCompleted?.(JSON.parse((ev as MessageEvent).data)); } catch { /* ignore */ }
    es.close();
  });
  es.addEventListener("job_cancelled", (ev) => {
    try { callbacks.onJobCancelled?.(JSON.parse((ev as MessageEvent).data)); } catch { /* ignore */ }
    es.close();
  });
  es.addEventListener("job_failed", (ev) => {
    try { callbacks.onJobFailed?.(JSON.parse((ev as MessageEvent).data)); } catch { /* ignore */ }
    es.close();
  });
  es.onerror = (err) => {
    callbacks.onError?.(err);
    // EventSource auto-reconnects unless we close it. The SSE endpoint sends
    // Last-Event-ID on reconnect, so the server replays missed events.
  };

  return { close: () => es.close() };
}

export async function cancelFolderUpload(token: string, jobId: string): Promise<{ job_id: string; status: string }> {
  return apiCall<{ job_id: string; status: string }>(
    `/documents/folder/${encodeURIComponent(jobId)}/cancel`,
    token,
    { method: "POST" },
  );
}

export async function getActiveFolderJob(token: string): Promise<FolderJobSnapshot | null> {
  return apiCall<FolderJobSnapshot | null>("/documents/folder/active", token);
}

export async function getFolderJob(token: string, jobId: string): Promise<FolderJobSnapshot> {
  return apiCall<FolderJobSnapshot>(`/documents/folder/${encodeURIComponent(jobId)}`, token);
}
