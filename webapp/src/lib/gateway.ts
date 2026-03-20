const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

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
}

export interface ChatMessage {
  id: string;
  role: string;
  content: string;
  mode_used: string | null;
  model_used?: string | null;
  reasoning_content?: string | null;
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
  conversationMode: boolean = false
): Promise<ChatResponse> {
  return apiCall<ChatResponse>("/chat", token, {
    method: "POST",
    body: JSON.stringify({
      message,
      mode,
      session_id: sessionId || null,
      conversation_mode: conversationMode,
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
  onSearchDone?: (sources: SearchSource[]) => void,
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
    }),
    signal,
  });

  if (!res.ok) {
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

  // Format A: server sends delta.reasoning_content (some API providers)
  let reasoningField = "";
  let reasoningFieldUsed = false;
  let reasoningFieldDone = false;

  // Format B/C: thinking in delta.content (mlx_vlm strips <think> to "")
  let rawContent = "";
  let firstContentSeen = false;
  let firstContentWasEmpty = false;

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

    if (ct !== undefined && ct !== null) {
      if (!firstContentSeen) {
        firstContentSeen = true;
        firstContentWasEmpty = ct === "";
      }
      if (ct) rawContent += ct;
      if (reasoningFieldUsed && ct.length > 0 && !reasoningFieldDone) {
        reasoningFieldDone = true;
      }
    }

    const assembled = synthesize();
    onChunk(assembled);
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

    // Format A: reasoning_content field present
    if (reasoningFieldUsed) {
      return (
        "<think>" +
        reasoningField +
        (reasoningFieldDone ? "</think>" : "") +
        rawContent
      );
    }

    // Format B: explicit <think> tags in content — pass through for parser
    if (rawContent.includes("<think>")) {
      return rawContent;
    }

    // Format C: mlx_vlm stripped <think> to "", but </think> appears literally
    const closeIdx = rawContent.indexOf("</think>");

    if (closeIdx !== -1 && firstContentWasEmpty) {
      const thinkText = rawContent.slice(0, closeIdx).trim();
      const respText = rawContent.slice(closeIdx + "</think>".length);
      return (thinkText ? "<think>" + thinkText + "</think>" : "") + respText;
    }

    if (firstContentWasEmpty && rawContent.trim() && closeIdx === -1) {
      return "<think>" + rawContent;
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
  file: File
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${GATEWAY_URL}/documents/upload`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Upload error ${res.status}: ${errText}`);
  }

  return res.json();
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
  file: File
): Promise<AttachResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(
    `${GATEWAY_URL}/documents/sessions/${sessionId}/attach`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    }
  );

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Attach error ${res.status}: ${errText}`);
  }

  return res.json();
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
