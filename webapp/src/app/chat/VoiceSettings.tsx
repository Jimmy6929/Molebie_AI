"use client";

import { KOKORO_VOICES, type VoicePreferences } from "@/lib/voice";

interface VoiceSettingsProps {
  open: boolean;
  settings: VoicePreferences;
  onChange: (next: VoicePreferences) => void;
}

export default function VoiceSettings({ open, settings, onChange }: VoiceSettingsProps) {
  if (!open) return null;

  return (
    <div className="absolute bottom-24 right-6 z-30 w-72 glass rounded-2xl p-3 border border-white/[0.08]">
      <div className="text-xs text-[#ddd] mb-2">Voice Settings</div>

      <label className="text-[11px] text-[#999] block mb-1">Voice</label>
      <select
        value={settings.voiceId}
        onChange={(e) => onChange({ ...settings, voiceId: e.target.value })}
        className="w-full bg-black/40 border border-white/[0.12] text-[#eaeaea] px-2 py-1.5 text-xs rounded-lg focus:outline-none focus:border-[#00ff41]/40"
      >
        {KOKORO_VOICES.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
          </option>
        ))}
      </select>

      <label className="text-[11px] text-[#999] block mt-3 mb-1">
        Speed: {settings.speed.toFixed(2)}x
      </label>
      <input
        type="range"
        min={0.5}
        max={2}
        step={0.05}
        value={settings.speed}
        onChange={(e) => onChange({ ...settings, speed: Number(e.target.value) })}
        className="w-full"
      />
    </div>
  );
}
