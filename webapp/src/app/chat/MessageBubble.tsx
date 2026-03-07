"use client";

import { useState, useEffect, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { parseThinkingContent } from "@/lib/thinkParser";

export interface MessageBubbleProps {
  role: "user" | "assistant" | "system";
  content: string;
  streaming?: boolean;
  mode?: string | null;
  model?: string | null;
  metadata?: Record<string, unknown>;
  streamStartedAt?: number;
}

function useElapsedSeconds(startedAt?: number, running?: boolean): number {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!startedAt || !running) return;
    setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [startedAt, running]);
  return elapsed;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

export default function MessageBubble({
  role,
  content,
  streaming = false,
  mode,
  streamStartedAt,
}: MessageBubbleProps) {
  const parsed = useMemo(() => parseThinkingContent(content), [content]);
  const [thinkOpen, setThinkOpen] = useState(false);
  const elapsed = useElapsedSeconds(streamStartedAt, streaming);

  // Auto-expand while the model is actively thinking, collapse when done
  useEffect(() => {
    if (parsed.isThinking) {
      setThinkOpen(true);
    } else if (!streaming && parsed.thinking) {
      setThinkOpen(false);
    }
  }, [parsed.isThinking, streaming, parsed.thinking]);

  const isUser = role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] text-sm ${
          isUser
            ? "bg-[#0d1f0d] border border-[#00ff41]/20 text-[#00ff41] px-3 py-2"
            : "bg-[#111111] border border-[#3a3a3a] text-[#b0b0b0] px-3 py-2"
        }`}
      >
        {/* Role / mode label */}
        <div className="text-[10px] text-[#77bb88] mb-1 flex items-center gap-2">
          <span>{isUser ? "> you" : "> ai"}</span>
          {mode && (
            <span
              className={`px-1 ${
                mode === "thinking"
                  ? "border border-[#ff9900]/40 text-[#ff9900]"
                  : "border border-[#3a3a3a] text-[#77bb88]"
              }`}
            >
              {mode}
            </span>
          )}
        </div>

        {/* Thinking panel (assistant only) */}
        {!isUser && parsed.thinking && (
          <div className="mb-2 border border-[#2a2a2a] bg-[#0a0a0a]">
            <button
              onClick={() => setThinkOpen(!thinkOpen)}
              className="w-full flex items-center gap-2 px-2 py-1 text-[10px] text-[#888] hover:text-[#aaa] transition-colors"
            >
              <span
                className={`transition-transform duration-200 ${
                  thinkOpen ? "rotate-90" : ""
                }`}
              >
                ▶
              </span>
              <span>
                {parsed.isThinking ? "thinking" : "thought process"}
              </span>
              {streamStartedAt && (
                <span className={`ml-auto tabular-nums ${parsed.isThinking ? "text-[#ff9900]" : "text-[#555]"}`}>
                  {formatElapsed(elapsed)}
                </span>
              )}
            </button>
            {thinkOpen && (
              <div className="px-2 pb-2 text-[11px] text-[#777] whitespace-pre-wrap leading-relaxed border-t border-[#2a2a2a]">
                {parsed.thinking}
              </div>
            )}
          </div>
        )}

        {/* Message body */}
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">
            {content}
            {streaming && (
              <span className="inline-block w-2 h-4 bg-[#00ff41] ml-0.5 animate-pulse" />
            )}
          </div>
        ) : (
          <div className="markdown-body break-words">
            {parsed.response ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    const codeStr = String(children).replace(/\n$/, "");
                    if (match) {
                      return (
                        <SyntaxHighlighter
                          style={oneDark}
                          language={match[1]}
                          PreTag="div"
                          customStyle={{
                            margin: 0,
                            background: "#050505",
                            border: "1px solid #3a3a3a",
                            borderRadius: "4px",
                            fontSize: "0.85em",
                          }}
                        >
                          {codeStr}
                        </SyntaxHighlighter>
                      );
                    }
                    return (
                      <code className="bg-[#1a1a1a] px-1 py-0.5 text-[#e0e0e0] text-[0.85em]" {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {parsed.response}
              </ReactMarkdown>
            ) : null}
            {streaming && (
              <span className="inline-block w-2 h-4 bg-[#00ff41] ml-0.5 animate-pulse" />
            )}
          </div>
        )}

        {/* Elapsed time footer */}
        {!isUser && streamStartedAt && elapsed > 0 && (
          <div className="text-[10px] text-[#555] mt-1.5 tabular-nums">
            {streaming ? `generating · ${formatElapsed(elapsed)}` : `${formatElapsed(elapsed)}`}
          </div>
        )}
      </div>
    </div>
  );
}
