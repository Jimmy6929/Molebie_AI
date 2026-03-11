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

const STOP_PHRASES = [
  "stop",
  "stop conversation",
  "end conversation",
  "goodbye",
  "bye",
  "bye bye",
  "that's all",
  "stop talking",
];

export function isStopCommand(text: string): boolean {
  const normalized = text.toLowerCase().trim().replace(/[.,!?]/g, "");
  return STOP_PHRASES.includes(normalized);
}

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
        voiceId: typeof parsed.voiceId === "string" ? parsed.voiceId : DEFAULT_VOICE_PREFERENCES.voiceId,
        speed: typeof parsed.speed === "number" ? parsed.speed : DEFAULT_VOICE_PREFERENCES.speed,
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
// ---------------------------------------------------------------------------

function createSilenceDetector(
  stream: MediaStream,
  onSilence: () => void,
  opts?: { threshold?: number; duration?: number }
) {
  const threshold = opts?.threshold ?? 12;
  const duration = opts?.duration ?? 2000;

  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  const buf = new Uint8Array(analyser.frequencyBinCount);
  let silenceStart: number | null = null;
  let raf: number | null = null;
  let stopped = false;

  function check() {
    if (stopped) return;
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
// Speech-to-Text via MediaRecorder + backend Whisper + silence detection
// ---------------------------------------------------------------------------

export function useSpeechRecognition(options?: {
  token?: string | null;
  onFinalTranscript?: (text: string) => void;
  autoStopOnSilence?: boolean;
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

        if (blob.size === 0) {
          setError("No audio recorded");
          return;
        }

        const token = tokenRef.current;
        if (!token) {
          setError("Not authenticated");
          return;
        }

        setIsTranscribing(true);
        try {
          const text = await transcribeAudio(token, blob);
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
        setError("Microphone permission denied. Allow access in browser settings.");
      } else {
        setError(err instanceof Error ? err.message : "Failed to start recording");
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
// Text-to-Speech via Kokoro TTS (local, through gateway)
// ---------------------------------------------------------------------------

export function useKokoroTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);

  const speak = useCallback(
    async (
      token: string,
      text: string,
      voice: string = "bm_george",
      speed: number = 1.0
    ): Promise<void> => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
      }
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }

      const clean = cleanSpeechText(text);
      if (!clean) return;

      setIsSpeaking(true);

      try {
        const blob = await fetchTTSAudio(token, clean, voice, speed);
        const url = URL.createObjectURL(blob);
        urlRef.current = url;
        const audio = new Audio(url);
        audioRef.current = audio;

        return new Promise<void>((resolve) => {
          audio.onended = () => {
            cleanup();
            resolve();
          };
          audio.onerror = () => {
            cleanup();
            resolve();
          };
          audio.play().catch(() => {
            cleanup();
            resolve();
          });
        });
      } catch {
        setIsSpeaking(false);
      }

      function cleanup() {
        if (urlRef.current) {
          URL.revokeObjectURL(urlRef.current);
          urlRef.current = null;
        }
        audioRef.current = null;
        setIsSpeaking(false);
      }
    },
    []
  );

  const cancel = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
    setIsSpeaking(false);
  }, []);

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  return { isSpeaking, speak, cancel };
}
