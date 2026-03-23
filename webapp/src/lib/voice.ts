"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudio, fetchTTSAudio } from "@/lib/gateway";

const VOICE_SETTINGS_KEY = "local-ai-voice-settings-v2";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VoicePreferences {
  voiceId: string;
  speed: number;
}

const DEFAULT_VOICE_PREFERENCES: VoicePreferences = {
  voiceId: "bm_george",
  speed: 1.0,
};

export const KOKORO_VOICES = [
  { id: "bm_george", label: "George (British Male)" },
  { id: "bm_lewis", label: "Lewis (British Male)" },
  { id: "bm_daniel", label: "Daniel (British Male)" },
  { id: "bm_fable", label: "Fable (British Male)" },
  { id: "bf_emma", label: "Emma (British Female)" },
  { id: "bf_isabella", label: "Isabella (British Female)" },
  { id: "bf_alice", label: "Alice (British Female)" },
  { id: "bf_lily", label: "Lily (British Female)" },
  { id: "am_michael", label: "Michael (American Male)" },
  { id: "am_adam", label: "Adam (American Male)" },
  { id: "af_sarah", label: "Sarah (American Female)" },
  { id: "af_nicole", label: "Nicole (American Female)" },
] as const;

// ---------------------------------------------------------------------------
// Stop / Wake-word detection
// ---------------------------------------------------------------------------

const STOP_PHRASES = [
  "stop",
  "stop conversation",
  "end conversation",
  "goodbye",
  "bye",
  "bye bye",
  "that's all",
  "stop talking",
  "end chat",
  "quit",
];

export function isStopCommand(text: string): boolean {
  const normalized = text.toLowerCase().trim().replace(/[.,!?]/g, "");
  return STOP_PHRASES.includes(normalized);
}

const WAKE_PHRASES = ["hey alfred", "hello alfred", "hi alfred", "alfred"];

export function extractWakeCommand(transcript: string): {
  isWakeWord: boolean;
  command: string;
} {
  const lower = transcript.toLowerCase().trim();
  for (const phrase of WAKE_PHRASES) {
    if (lower === phrase || lower.startsWith(phrase + " ") || lower.startsWith(phrase + ",")) {
      const command = transcript
        .slice(phrase.length)
        .replace(/^[,.\s]+/, "")
        .trim();
      return { isWakeWord: true, command };
    }
  }
  return { isWakeWord: false, command: "" };
}

// ---------------------------------------------------------------------------
// Text cleanup for TTS
// ---------------------------------------------------------------------------

function cleanSpeechText(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, " ")
    .replace(/<\/think>/g, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[#*_>\-\[\]()]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 2);
}

// ---------------------------------------------------------------------------
// Voice Settings (localStorage)
// ---------------------------------------------------------------------------

export function useVoiceSettings() {
  const [settings, setSettings] = useState<VoicePreferences>(() => {
    if (typeof window === "undefined") return DEFAULT_VOICE_PREFERENCES;
    try {
      const raw = window.localStorage.getItem(VOICE_SETTINGS_KEY);
      if (!raw) return DEFAULT_VOICE_PREFERENCES;
      const parsed = JSON.parse(raw) as Partial<VoicePreferences>;
      return {
        voiceId:
          typeof parsed.voiceId === "string"
            ? parsed.voiceId
            : DEFAULT_VOICE_PREFERENCES.voiceId,
        speed:
          typeof parsed.speed === "number"
            ? parsed.speed
            : DEFAULT_VOICE_PREFERENCES.speed,
      };
    } catch {
      return DEFAULT_VOICE_PREFERENCES;
    }
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(VOICE_SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  return { settings, setSettings };
}

// ---------------------------------------------------------------------------
// Silence detection helper
// startDelay: don't fire for first N ms — gives the user time to start speaking
// ---------------------------------------------------------------------------

export function createSilenceDetector(
  stream: MediaStream,
  onSilence: () => void,
  opts?: { threshold?: number; duration?: number; startDelay?: number }
) {
  const threshold = opts?.threshold ?? 15;
  const duration = opts?.duration ?? 2000;
  const startDelay = opts?.startDelay ?? 2000; // 2s before silence can fire

  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  const buf = new Uint8Array(analyser.frequencyBinCount);
  let silenceStart: number | null = null;
  let raf: number | null = null;
  let stopped = false;
  const startTime = Date.now();

  function check() {
    if (stopped) return;

    // Don't allow silence to trigger in the first startDelay ms
    if (Date.now() - startTime < startDelay) {
      raf = requestAnimationFrame(check);
      return;
    }

    analyser.getByteFrequencyData(buf);
    const avg = buf.reduce((s, v) => s + v, 0) / buf.length;

    if (avg < threshold) {
      if (silenceStart === null) silenceStart = Date.now();
      else if (Date.now() - silenceStart > duration) {
        onSilence();
        return;
      }
    } else {
      silenceStart = null;
    }
    raf = requestAnimationFrame(check);
  }

  check();

  return () => {
    stopped = true;
    if (raf !== null) cancelAnimationFrame(raf);
    source.disconnect();
    ctx.close().catch(() => {});
  };
}

// ---------------------------------------------------------------------------
// Speech-to-Text via MediaRecorder + backend Whisper
// ---------------------------------------------------------------------------

export function useSpeechRecognition(options?: {
  token?: string | null;
  onFinalTranscript?: (text: string) => void;
  autoStopOnSilence?: boolean;
  verifySpeaker?: boolean;
}) {
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const abortedRef = useRef(false);
  const silenceCleanupRef = useRef<(() => void) | null>(null);

  const onFinalTranscriptRef = useRef(options?.onFinalTranscript);
  onFinalTranscriptRef.current = options?.onFinalTranscript;
  const tokenRef = useRef(options?.token);
  tokenRef.current = options?.token;
  const autoStopRef = useRef(options?.autoStopOnSilence ?? false);
  autoStopRef.current = options?.autoStopOnSilence ?? false;
  const verifySpeakerRef = useRef(options?.verifySpeaker ?? false);
  verifySpeakerRef.current = options?.verifySpeaker ?? false;

  const supportsVoiceInput =
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function";

  const stopListening = useCallback(() => {
    abortedRef.current = false;
    silenceCleanupRef.current?.();
    silenceCleanupRef.current = null;
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;
    setIsListening(false);
  }, []);

  const startListening = useCallback(async () => {
    setError(null);
    setTranscript("");
    abortedRef.current = false;

    if (!supportsVoiceInput) {
      setError("Microphone access is not available in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "audio/mp4";

      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        silenceCleanupRef.current?.();
        silenceCleanupRef.current = null;

        if (abortedRef.current) {
          chunksRef.current = [];
          return;
        }

        const blob = new Blob(chunksRef.current, { type: mimeType });
        chunksRef.current = [];

        if (blob.size === 0) return;

        const token = tokenRef.current;
        if (!token) {
          setError("Not authenticated");
          return;
        }

        setIsTranscribing(true);
        try {
          const result = await transcribeAudio(
            token,
            blob,
            verifySpeakerRef.current
          );
          if (verifySpeakerRef.current && result.speaker_verified === false) {
            // Not the enrolled speaker — silently discard
            return;
          }
          const text = result.text;
          setTranscript(text);
          if (text.trim() && onFinalTranscriptRef.current) {
            onFinalTranscriptRef.current(text.trim());
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : "Transcription failed");
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsListening(true);

      if (autoStopRef.current) {
        silenceCleanupRef.current = createSilenceDetector(stream, () => {
          stopListening();
        });
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError(
          "Microphone permission denied. Allow access in browser settings."
        );
      } else {
        setError(
          err instanceof Error ? err.message : "Failed to start recording"
        );
      }
    }
  }, [supportsVoiceInput, stopListening]);

  const abortListening = useCallback(() => {
    abortedRef.current = true;
    silenceCleanupRef.current?.();
    silenceCleanupRef.current = null;
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    chunksRef.current = [];
    setIsListening(false);
  }, []);

  useEffect(() => {
    return () => {
      silenceCleanupRef.current?.();
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    supportsSpeechRecognition: supportsVoiceInput,
    isListening,
    isTranscribing,
    transcript,
    error,
    startListening,
    stopListening,
    abortListening,
  };
}

// ---------------------------------------------------------------------------
// Text-to-Speech via Kokoro TTS
// ---------------------------------------------------------------------------

function playAudioBlob(blob: Blob): Promise<void> {
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  return new Promise<void>((resolve) => {
    const done = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.onended = done;
    audio.onerror = done;
    audio.play().catch(done);
  });
}

// ---------------------------------------------------------------------------
// StreamingSpeaker — starts TTS during LLM streaming, not after
// ---------------------------------------------------------------------------

export interface StreamingSpeaker {
  /** Call on every SSE chunk with the accumulated content so far. */
  feed(accumulatedContent: string): void;
  /** Call when the SSE stream ends — flushes the last partial sentence. */
  finish(): void;
  /** Abort all pending TTS and stop playback. */
  cancel(): void;
  /** Resolves when all playback is complete (or cancelled). */
  done: Promise<void>;
}

function createStreamingSpeakerImpl(
  token: string,
  voice: string,
  speed: number,
  setIsSpeaking: (v: boolean) => void,
): StreamingSpeaker {
  // ── Sentence queue with resolve-based signaling ──────────────────────
  const queue: string[] = [];
  let notifyResolve: (() => void) | null = null;
  let cancelled = false;
  let finished = false;

  function enqueue(sentence: string) {
    queue.push(sentence);
    if (notifyResolve) { notifyResolve(); notifyResolve = null; }
  }

  function waitForItem(): Promise<void> {
    if (queue.length > 0 || finished || cancelled) return Promise.resolve();
    return new Promise<void>((r) => { notifyResolve = r; });
  }

  // ── Sentence detection state ─────────────────────────────────────────
  let processedLength = 0;
  let uncommitted = "";

  function feed(accumulatedContent: string) {
    if (cancelled || finished) return;

    const cleaned = cleanSpeechText(accumulatedContent);
    if (cleaned.length <= processedLength) return;

    const newText = cleaned.slice(processedLength);
    processedLength = cleaned.length;
    uncommitted += newText;

    // Split on sentence-ending punctuation followed by whitespace
    const segments = uncommitted.split(/(?<=[.!?])\s+/);
    if (segments.length > 1) {
      // All but last segment are complete sentences
      for (let i = 0; i < segments.length - 1; i++) {
        const s = segments[i].trim();
        if (s.length > 2) enqueue(s);
      }
      uncommitted = segments[segments.length - 1];
    }
  }

  function finish() {
    if (cancelled || finished) return;
    finished = true;
    // Flush remaining text as final sentence
    const remaining = uncommitted.trim();
    if (remaining.length > 2) enqueue(remaining);
    uncommitted = "";
    // Wake consumer if waiting
    if (notifyResolve) { notifyResolve(); notifyResolve = null; }
  }

  function cancel() {
    cancelled = true;
    finished = true;
    if (notifyResolve) { notifyResolve(); notifyResolve = null; }
  }

  // ── Playback loop with one-ahead prefetch ────────────────────────────
  const done = (async () => {
    let prefetch: Promise<Blob | null> | null = null;
    let started = false;

    while (true) {
      if (cancelled) break;

      let blob: Blob | null;

      if (prefetch) {
        // Use the audio we already started fetching
        blob = await prefetch;
        prefetch = null;
      } else {
        // Wait for a sentence to arrive
        await waitForItem();
        if (cancelled) break;
        if (queue.length === 0 && finished) break;
        if (queue.length === 0) continue;

        const sentence = queue.shift()!;
        blob = await fetchTTSAudio(token, sentence, voice, speed).catch(() => null);
      }

      if (!blob || cancelled) break;

      // Prefetch next sentence while current one plays
      if (queue.length > 0) {
        const next = queue.shift()!;
        prefetch = fetchTTSAudio(token, next, voice, speed).catch(() => null);
      }

      if (!started) { setIsSpeaking(true); started = true; }
      await playAudioBlob(blob);
    }

    setIsSpeaking(false);
  })();

  return { feed, finish, cancel, done };
}

// ---------------------------------------------------------------------------
// Hook: useKokoroTTS
// ---------------------------------------------------------------------------

export function useKokoroTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const cancelledRef = useRef(false);
  const activeStreamCancelRef = useRef<(() => void) | null>(null);

  // Legacy: speak a complete text (non-streaming, sentence-parallel)
  const speak = useCallback(
    async (
      token: string,
      text: string,
      voice: string = "bm_george",
      speed: number = 1.0
    ): Promise<void> => {
      const clean = cleanSpeechText(text);
      if (!clean) return;

      cancelledRef.current = false;
      setIsSpeaking(true);

      try {
        const sentences = splitSentences(clean);
        if (sentences.length === 0) return;

        const audioPromises = sentences.map((s) =>
          fetchTTSAudio(token, s, voice, speed).catch(() => null)
        );

        for (const promise of audioPromises) {
          if (cancelledRef.current) break;
          const blob = await promise;
          if (!blob || cancelledRef.current) break;
          await playAudioBlob(blob);
        }
      } catch {
        // TTS failed silently — conversation loop continues
      } finally {
        setIsSpeaking(false);
      }
    },
    []
  );

  // Streaming: start TTS during LLM generation
  const createStreamingSpeaker = useCallback(
    (token: string, voice: string = "bm_george", speed: number = 1.0): StreamingSpeaker => {
      // Cancel any previous streaming speaker
      activeStreamCancelRef.current?.();

      const speaker = createStreamingSpeakerImpl(token, voice, speed, setIsSpeaking);
      activeStreamCancelRef.current = speaker.cancel;

      // Clear ref when playback finishes naturally
      speaker.done.then(() => {
        if (activeStreamCancelRef.current === speaker.cancel) {
          activeStreamCancelRef.current = null;
        }
      });

      return speaker;
    },
    []
  );

  const cancel = useCallback(() => {
    // Cancel legacy speak
    cancelledRef.current = true;
    // Cancel streaming speaker
    activeStreamCancelRef.current?.();
    activeStreamCancelRef.current = null;
    setIsSpeaking(false);
  }, []);

  return { isSpeaking, speak, createStreamingSpeaker, cancel };
}
