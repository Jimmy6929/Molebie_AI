# API Reference

## Health Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Basic health check |
| `/health/auth` | GET | Yes | Validates JWT, returns user info |
| `/health/inference` | GET | No | Instant + thinking tier status |

## Chat Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat` | POST | Yes | Send message, receive full response |
| `/chat/stream` | POST | Yes | Send message, receive SSE stream |
| `/chat/sessions` | GET | Yes | List chat sessions |
| `/chat/sessions/create` | POST | Yes | Create empty session |
| `/chat/sessions/{id}/messages` | GET | Yes | Get session messages |
| `/chat/sessions/{id}` | PATCH | Yes | Rename session |
| `/chat/sessions/{id}` | DELETE | Yes | Delete session |

## Document Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/documents/upload` | POST | Yes | Upload document for RAG |
| `/documents` | GET | Yes | List documents |
| `/documents/{id}` | DELETE | Yes | Delete document |
| `/documents/sessions/{id}/attach` | POST | Yes | Attach doc to session |

## Voice Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat/transcribe` | POST | Yes | Speech-to-text (Whisper) |
| `/chat/tts` | POST | Yes | Text-to-speech (Kokoro, returns MP3) |
| `/chat/voice-enroll` | POST | Yes | Enroll voice sample |
| `/chat/voice-profile` | GET/DELETE | Yes | Manage voice profile |

## Chat Request/Response

```json
// POST /chat or /chat/stream
{
  "session_id": "uuid | null",
  "message": "string",
  "mode": "instant | thinking | thinking_harder",
  "conversation_mode": false,
  "image": "data:image/jpeg;base64,... | null"
}
```

## Database Schema

| Table | Description |
|-------|-------------|
| `profiles` | User profiles (auto-created on signup) |
| `chat_sessions` | Conversations with pinning support |
| `chat_messages` | Messages with `reasoning_content` and `mode_used` |
| `message_images` | Image attachments (metadata; files in local storage) |
| `documents` | Uploaded document metadata |
| `document_chunks` | Chunks with sqlite-vec embeddings + FTS5 for BM25 |
| `session_documents` | Per-session document attachments |

All queries enforce user isolation — users can only access their own data.
