"use client";

import { KOKORO_VOICES, type VoicePreferences } from "@/lib/voice";
import { useEffect } from "react";

interface VoiceSettingsProps {
  open: boolean;
  settings: VoicePreferences;
  onChange: (next: VoicePreferences) => void;
  onClose?: () => void;
}

export default function VoiceSettings({
  open,
  settings,
  onChange,
  onClose,
}: VoiceSettingsProps) {
  useEffect(() => {
    if (!open || !onClose) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-x-2 bottom-20 z-30 md:absolute md:inset-auto md:bottom-24 md:right-6 md:w-80 glass rounded-2xl p-4 border border-white/[0.08] space-y-4">
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
    </div>
  );
}
