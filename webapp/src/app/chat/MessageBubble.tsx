"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { parseThinkingContent } from "@/lib/thinkParser";
import type { SearchSource } from "@/lib/gateway";

export interface MessageBubbleProps {
  role: "user" | "assistant" | "system";
  content: string;
  streaming?: boolean;
  mode?: string | null;
  model?: string | null;
  metadata?: Record<string, unknown>;
  streamStartedAt?: number;
  sources?: SearchSource[];
  isSearching?: boolean;
  imageUrl?: string | null;
  onRegenerate?: () => void;
}

// ── CodeBlock with copy button ───────────────────────────────────────────

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  return (
    <div className="relative group my-2">
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#0a0a0a] border border-white/[0.06] border-b-0 rounded-t-xl text-[10px]">
        <span className="text-[#888] font-medium">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[#888] hover:text-[#ccc] transition-colors"
        >
          {copied ? (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Copied</span>
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language}
        PreTag="div"
        customStyle={{
          margin: 0,
          background: "#050505",
          border: "1px solid rgba(255,255,255,0.06)",
          borderTop: "none",
          borderRadius: "0 0 12px 12px",
          fontSize: "0.85em",
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

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

/** Inject inline citation superscripts: [1] → clickable badge linking to source */
function injectCitations(text: string, sources?: SearchSource[]): string {
  if (!sources || sources.length === 0) return text;
  return text.replace(/\[(\d+)\]/g, (match, numStr) => {
    const idx = parseInt(numStr, 10) - 1;
    if (idx < 0 || idx >= sources.length) return match;
    const src = sources[idx];
    return `[<sup>${numStr}</sup>](${src.url} "${(src.title || src.url).replace(/"/g, "'")}")`;
  });
}

// ── Main component ──────────────────────────────────────────────────────

export default function MessageBubble({
  role,
  content,
  streaming = false,
  mode,
  streamStartedAt,
  sources,
  isSearching,
  imageUrl,
  onRegenerate,
}: MessageBubbleProps) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [msgCopied, setMsgCopied] = useState(false);
  const parsed = useMemo(() => parseThinkingContent(content), [content]);
  const [thinkOpen, setThinkOpen] = useState(false);
  const elapsed = useElapsedSeconds(streamStartedAt, streaming);

  // Inject citation links into response text
  const renderedResponse = useMemo(
    () => (parsed.response ? injectCitations(parsed.response, sources) : null),
    [parsed.response, sources],
  );

  useEffect(() => {
    if (parsed.isThinking) {
      setThinkOpen(true);
    } else if (!streaming && parsed.thinking) {
      setThinkOpen(false);
    }
  }, [parsed.isThinking, streaming, parsed.thinking]);

  const handleCopyMessage = useCallback(() => {
    navigator.clipboard.writeText(parsed.response || content).then(() => {
      setMsgCopied(true);
      setTimeout(() => setMsgCopied(false), 2000);
    });
  }, [parsed.response, content]);

  const isUser = role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] text-sm rounded-2xl transition-all ${
          isUser
            ? "bg-[#00ff41]/[0.1] text-[#00ff41] px-4 py-3"
            : "glass-surface px-4 py-3 text-[#dddddd]"
        }`}
      >
        {/* Role label */}
        <div className="text-[10px] text-[#aaa] mb-1.5 flex items-center gap-2">
          <span className="font-medium">{isUser ? "You" : "AI"}</span>
          {mode && (
            <span
              className={`px-1.5 py-0.5 rounded-md text-[9px] ${
                mode === "thinking" || mode === "thinking_harder"
                  ? "bg-[#ffbb33]/15 text-[#ffcc33]"
                  : "bg-white/[0.06] text-[#aaa]"
              }`}
            >
              {mode === "thinking_harder" ? "Think+" : mode === "thinking" ? "Think" : "Fast"}
            </span>
          )}
          {!isUser && sources && sources.length > 0 && (
            <span className="px-1.5 py-0.5 rounded-md text-[9px] bg-[#3399ff]/15 text-[#66bbff]">
              Web
            </span>
          )}
        </div>

        {/* Thinking panel */}
        {!isUser && parsed.thinking && (
          <div className="mb-2 rounded-xl bg-black/30 overflow-hidden">
            <button
              onClick={() => setThinkOpen(!thinkOpen)}
              className="w-full flex items-center gap-2 px-3 py-2 text-[10px] text-[#aaa] hover:text-[#ccc] transition-colors"
            >
              <span
                className={`transition-transform duration-200 text-[8px] ${
                  thinkOpen ? "rotate-90" : ""
                }`}
              >
                ▶
              </span>
              <span>
                {parsed.isThinking ? "Thinking..." : "Thought process"}
              </span>
              {streamStartedAt && (
                <span className={`ml-auto tabular-nums ${parsed.isThinking ? "text-[#ffcc33]" : "text-[#888]"}`}>
                  {formatElapsed(elapsed)}
                </span>
              )}
            </button>
            {thinkOpen && (
              <div className="px-3 pb-2.5 text-[11px] text-[#aaa] whitespace-pre-wrap leading-relaxed border-t border-white/[0.06]">
                {parsed.thinking}
              </div>
            )}
          </div>
        )}

        {/* Searching indicator */}
        {!isUser && isSearching && (
          <div className="mb-2 flex items-center gap-2 text-[11px] text-[#66bbff]">
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
              <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
            </svg>
            Searching the web...
          </div>
        )}

        {/* Attached image */}
        {isUser && imageUrl && (
          <div className="mb-2">
            <img
              src={imageUrl}
              alt="Attached"
              className="max-w-xs max-h-64 rounded-xl border border-white/[0.06] cursor-pointer hover:opacity-90 transition-opacity"
              onClick={() => window.open(imageUrl, "_blank")}
            />
          </div>
        )}

        {/* Message body */}
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">
            {content}
            {streaming && (
              <span className="inline-block w-2 h-4 bg-[#00ff41] ml-0.5 animate-pulse rounded-sm" />
            )}
          </div>
        ) : (
          <div className="markdown-body break-words">
            {renderedResponse ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    const codeStr = String(children).replace(/\n$/, "");
                    if (match) {
                      return <CodeBlock language={match[1]} code={codeStr} />;
                    }
                    return (
                      <code className="bg-white/[0.08] px-1.5 py-0.5 rounded-md text-[#f0f0f0] text-[0.85em]" {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {renderedResponse}
              </ReactMarkdown>
            ) : null}
            {streaming && (
              <span className="inline-block w-2 h-4 bg-[#00ff41] ml-0.5 animate-pulse rounded-sm" />
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && sources && sources.length > 0 && (
          <div className="mt-2 rounded-xl bg-black/30 overflow-hidden">
            <button
              onClick={() => setSourcesOpen(!sourcesOpen)}
              className="w-full flex items-center gap-2 px-3 py-2 text-[10px] text-[#66bbff] hover:text-[#99ccff] transition-colors"
            >
              <span
                className={`transition-transform duration-200 text-[8px] ${
                  sourcesOpen ? "rotate-90" : ""
                }`}
              >
                ▶
              </span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <span>{sources.length} source{sources.length !== 1 ? "s" : ""}</span>
            </button>
            {sourcesOpen && (
              <div className="px-3 pb-2.5 space-y-1.5 border-t border-white/[0.06]">
                {sources.map((src, i) => (
                  <a
                    key={i}
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-[11px] text-[#66bbff] hover:text-[#99ccff] truncate transition-colors"
                    title={src.url}
                  >
                    {src.title || src.url}
                  </a>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Elapsed time footer */}
        {!isUser && streamStartedAt && elapsed > 0 && (
          <div className="text-[10px] text-[#999] mt-2 tabular-nums">
            {streaming ? `Generating · ${formatElapsed(elapsed)}` : `${formatElapsed(elapsed)}`}
          </div>
        )}

        {/* Action bar: Copy + Regenerate */}
        {!isUser && !streaming && (parsed.response || content) && (
          <div className="flex items-center gap-3 mt-2 pt-1.5 border-t border-white/[0.04]">
            <button
              onClick={handleCopyMessage}
              className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#ccc] transition-colors"
              title="Copy message"
            >
              {msgCopied ? (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  Copied
                </>
              ) : (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  Copy
                </>
              )}
            </button>
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                className="flex items-center gap-1 text-[10px] text-[#888] hover:text-[#ccc] transition-colors"
                title="Regenerate response"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="1 4 1 10 7 10" />
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                </svg>
                Regenerate
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
