"use client";

import { useState } from "react";

import type { FolderJobStatus } from "@/lib/gateway";

interface ErrorEntry {
  relative_path: string;
  error: string;
}

interface FolderUploadProgressProps {
  rootLabel: string;
  status: FolderJobStatus | "uploading";
  totalFiles: number;
  totalBytes: number;
  processedFiles: number;
  failedFiles: number;
  skippedFiles: number;
  processedBytes: number;
  currentFilename: string | null;
  errors: ErrorEntry[];
  onCancel: () => void;
  onDismiss: () => void;
}

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function FolderUploadProgress({
  rootLabel,
  status,
  totalFiles,
  totalBytes,
  processedFiles,
  failedFiles,
  skippedFiles,
  processedBytes,
  currentFilename,
  errors,
  onCancel,
  onDismiss,
}: FolderUploadProgressProps) {
  const [errorsOpen, setErrorsOpen] = useState(false);

  const done = processedFiles + failedFiles + skippedFiles;
  const pct =
    totalBytes > 0
      ? Math.min(100, Math.round((processedBytes / totalBytes) * 100))
      : totalFiles > 0
      ? Math.min(100, Math.round((done / totalFiles) * 100))
      : 0;

  const isTerminal =
    status === "completed" || status === "failed" || status === "cancelled";

  const barClass =
    status === "completed"
      ? "bg-[#00ff41]"
      : status === "failed"
      ? "bg-[#ff4444]"
      : status === "cancelled"
      ? "bg-[#888]"
      : "bg-[#33ccff]";

  const statusLabel =
    status === "completed"
      ? "✓ Done"
      : status === "failed"
      ? "✗ Failed"
      : status === "cancelled"
      ? "⨯ Cancelled"
      : status === "uploading"
      ? "↑ Uploading"
      : "● Processing";

  return (
    <div className="mb-2 glass rounded-xl px-3 py-2 animate-fade-in space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-[#ccc] font-medium truncate flex-1">
          📁 {rootLabel}
        </span>
        <span
          className={`text-[10px] shrink-0 ${
            status === "completed"
              ? "text-[#00ff41]"
              : status === "failed"
              ? "text-[#ff4444]"
              : status === "cancelled"
              ? "text-[#888]"
              : "text-[#66ddff]"
          }`}
        >
          {statusLabel}
        </span>
      </div>

      <div className="flex items-center justify-between text-[10px] text-[#888]">
        <span>
          {done} / {totalFiles} files
          {failedFiles > 0 && (
            <span className="text-[#ff4444]"> · {failedFiles} failed</span>
          )}
          {skippedFiles > 0 && (
            <span className="text-[#888]"> · {skippedFiles} skipped</span>
          )}
        </span>
        <span>
          {fmtBytes(processedBytes)} / {fmtBytes(totalBytes)} · {pct}%
        </span>
      </div>

      <div className="w-full h-1 rounded-full overflow-hidden bg-white/[0.06]">
        <div
          className={`h-full ${barClass} rounded-full transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {currentFilename && !isTerminal && (
        <p className="text-[10px] text-[#888] truncate">
          <span className="text-[#66ddff]">→</span> {currentFilename}
        </p>
      )}

      {errors.length > 0 && (
        <div className="text-[10px]">
          <button
            type="button"
            onClick={() => setErrorsOpen((v) => !v)}
            className="text-[#ff4444] hover:text-[#ff7777] transition-colors"
          >
            {errorsOpen ? "▼" : "▶"} {errors.length} error{errors.length === 1 ? "" : "s"}
          </button>
          {errorsOpen && (
            <ul className="mt-1 space-y-0.5 max-h-24 overflow-y-auto pl-3">
              {errors.map((e, i) => (
                <li key={`${e.relative_path}-${i}`} className="text-[#aaa] truncate" title={e.error}>
                  <span className="text-[#ccc]">{e.relative_path}</span>
                  <span className="text-[#666]"> — {e.error}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="flex items-center justify-end gap-2 pt-0.5">
        {!isTerminal && (
          <button
            type="button"
            onClick={onCancel}
            className="text-[10px] px-2 py-0.5 rounded-md bg-[#ff4444]/15 text-[#ff7777] hover:bg-[#ff4444]/25 transition-all"
          >
            Cancel
          </button>
        )}
        {isTerminal && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-[10px] px-2 py-0.5 rounded-md bg-white/[0.06] text-[#aaa] hover:bg-white/[0.12] transition-all"
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
