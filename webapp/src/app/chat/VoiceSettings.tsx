"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { KOKORO_VOICES, type VoicePreferences } from "@/lib/voice";
import {
  enrollVoiceSample,
  getVoiceProfileStatus,
  deleteVoiceProfile,
  type VoiceProfileStatus,
} from "@/lib/gateway";

interface VoiceSettingsProps {
  open: boolean;
  token: string | null;
  settings: VoicePreferences;
  onChange: (next: VoicePreferences) => void;
  wakeWordEnabled: boolean;
  onWakeWordToggle: (on: boolean) => void;
  speakerVerifyEnabled: boolean;
  onSpeakerVerifyToggle: (on: boolean) => void;
  /** Called before enrollment starts so the conversation mic can be paused */
  onEnrollStart: () => void;
  /** Called after enrollment finishes so the conversation mic can resume */
  onEnrollEnd: () => void;
}

export default function VoiceSettings({
  open,
  token,
  settings,
  onChange,
  wakeWordEnabled,
  onWakeWordToggle,
  speakerVerifyEnabled,
  onSpeakerVerifyToggle,
  onEnrollStart,
  onEnrollEnd,
}: VoiceSettingsProps) {
  const [profile, setProfile] = useState<VoiceProfileStatus | null>(null);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollCountdown, setEnrollCountdown] = useState(0);
  const [enrollError, setEnrollError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadProfile = useCallback(async () => {
    if (!token) return;
    try {
      const status = await getVoiceProfileStatus(token);
      setProfile(status);
    } catch {
      setProfile(null);
    }
  }, [token]);

  useEffect(() => {
    if (open && token) loadProfile();
  }, [open, token, loadProfile]);

  async function handleEnrollSample() {
    if (!token || enrolling) return;
    setEnrollError(null);
    setEnrolling(true);
    setEnrollCountdown(5);

    // Pause the conversation mic so there's no conflict
    onEnrollStart();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorderRef.current = recorder;

      // Countdown display
      countdownRef.current = setInterval(() => {
        setEnrollCountdown((n) => {
          if (n <= 1) {
            if (countdownRef.current) clearInterval(countdownRef.current);
            return 0;
          }
          return n - 1;
        });
      }, 1000);

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (countdownRef.current) clearInterval(countdownRef.current);
        const blob = new Blob(chunksRef.current, { type: mimeType });
        chunksRef.current = [];

        if (blob.size === 0) {
          setEnrollError("No audio captured — try again");
          setEnrolling(false);
          onEnrollEnd();
          return;
        }

        try {
          const result = await enrollVoiceSample(token!, blob);
          setProfile(result);
        } catch (err) {
          setEnrollError(err instanceof Error ? err.message : "Enrollment failed");
        } finally {
          setEnrolling(false);
          setEnrollCountdown(0);
          onEnrollEnd();
        }
      };

      recorder.start();
      setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, 5000);
    } catch (err) {
      if (countdownRef.current) clearInterval(countdownRef.current);
      setEnrollError(err instanceof Error ? err.message : "Mic access denied");
      setEnrolling(false);
      setEnrollCountdown(0);
      onEnrollEnd();
    }
  }

  async function handleDeleteProfile() {
    if (!token) return;
    try {
      await deleteVoiceProfile(token);
      setProfile({ enrolled: false, n_samples: 0, complete: false, required: 3 });
      onSpeakerVerifyToggle(false);
    } catch {
      // ignore
    }
  }

  if (!open) return null;

  const sampleLabel = (() => {
    if (enrolling) return `Recording... ${enrollCountdown}s`;
    if (profile?.complete) return "Add another sample";
    return `Record sample ${(profile?.n_samples ?? 0) + 1} of ${profile?.required ?? 3}`;
  })();

  return (
    <div className="absolute bottom-24 right-6 z-30 w-80 glass rounded-2xl p-4 border border-white/[0.08] space-y-4">
      <div className="text-xs font-medium text-[#ddd]">Chat Voice Settings</div>

      {/* Voice select */}
      <div>
        <label className="text-[11px] text-[#999] block mb-1">Voice</label>
        <select
          value={settings.voiceId}
          onChange={(e) => onChange({ ...settings, voiceId: e.target.value })}
          className="w-full bg-black/40 border border-white/[0.12] text-[#eaeaea] px-2 py-1.5 text-xs rounded-lg focus:outline-none focus:border-[#00ff41]/40"
        >
          {KOKORO_VOICES.map((v) => (
            <option key={v.id} value={v.id}>{v.label}</option>
          ))}
        </select>
      </div>

      {/* Speed */}
      <div>
        <label className="text-[11px] text-[#999] block mb-1">
          Speed: {settings.speed.toFixed(2)}x
        </label>
        <input
          type="range" min={0.5} max={2} step={0.05}
          value={settings.speed}
          onChange={(e) => onChange({ ...settings, speed: Number(e.target.value) })}
          className="w-full"
        />
      </div>

      <hr className="border-white/[0.06]" />

      {/* Wake word */}
      <label className="flex items-center gap-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={wakeWordEnabled}
          onChange={(e) => onWakeWordToggle(e.target.checked)}
          className="accent-[#33ccff]"
        />
        <div>
          <div className="text-[11px] text-[#ccc]">&quot;Hey Chat&quot; wake word</div>
          <div className="text-[10px] text-[#666]">Always listen, only respond to trigger phrase</div>
        </div>
      </label>

      <hr className="border-white/[0.06]" />

      {/* Speaker verification */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={speakerVerifyEnabled}
            onChange={(e) => onSpeakerVerifyToggle(e.target.checked)}
            disabled={!profile?.complete}
            className="accent-[#33ccff] disabled:opacity-40"
          />
          <div>
            <div className="text-[11px] text-[#ccc]">Only respond to my voice</div>
            <div className="text-[10px] text-[#666]">
              {!profile?.complete ? "Enroll 3 samples first to enable" : "Voice fingerprint active"}
            </div>
          </div>
        </label>

        {/* Progress */}
        <div className="flex gap-1">
          {[1, 2, 3].map((n) => (
            <div
              key={n}
              className={`h-1 flex-1 rounded-full transition-all ${
                (profile?.n_samples ?? 0) >= n ? "bg-[#33ccff]" : "bg-white/[0.08]"
              }`}
            />
          ))}
        </div>
        <div className="text-[10px] text-[#888]">
          {profile?.complete
            ? "Voice enrolled ✓"
            : profile?.enrolled
              ? `${profile.n_samples}/${profile.required} samples recorded`
              : "No voice profile — record 3 samples to enable"}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleEnrollSample}
            disabled={enrolling}
            className={`flex-1 text-[10px] px-2.5 py-1.5 rounded-lg border transition-all ${
              enrolling
                ? "bg-[#ff4444]/15 text-[#ff6666] border-[#ff4444]/30"
                : "bg-[#33ccff]/15 text-[#66ddff] border-[#33ccff]/30 hover:bg-[#33ccff]/25"
            } disabled:cursor-wait`}
          >
            {sampleLabel}
          </button>
          {profile?.enrolled && (
            <button
              onClick={handleDeleteProfile}
              className="text-[10px] px-2.5 py-1.5 rounded-lg text-[#ff7777] border border-[#ff4444]/30 hover:bg-[#ff4444]/10 transition-all"
            >
              Reset
            </button>
          )}
        </div>

        {enrollError && (
          <div className="text-[10px] text-[#ff7777]">{enrollError}</div>
        )}
      </div>
    </div>
  );
}
