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
  mode: "instant" | "thinking" = "instant",
  sessionId?: string
): Promise<ChatResponse> {
  return apiCall<ChatResponse>("/chat", token, {
    method: "POST",
    body: JSON.stringify({
      message,
      mode,
      session_id: sessionId || null,
    }),
  });
}

export async function sendMessageStream(
  token: string,
  message: string,
  mode: "instant" | "thinking" = "instant",
  sessionId?: string,
  onChunk: (text: string) => void = () => {},
  onSessionId: (id: string) => void = () => {},
  signal?: AbortSignal
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
  let fullContent = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    const lines = text.split("\n");

    for (const line of lines) {
      if (line.startsWith("data: ") && !line.includes("[DONE]")) {
        try {
          const data = JSON.parse(line.slice(6));

          // First event carries session_id metadata
          if (data.session_id && !data.choices) {
            onSessionId(data.session_id);
            continue;
          }

          const content = data.choices?.[0]?.delta?.content;
          if (content) {
            fullContent += content;
            onChunk(fullContent);
          }
        } catch {
          // skip parse errors
        }
      }
    }
  }

  return fullContent;
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

export async function checkInferenceHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(`${GATEWAY_URL}/health/inference`);
  return res.json();
}
