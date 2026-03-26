"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  login,
  register,
  loginSimple,
  getAuthMode,
  type AuthMode,
} from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const [authMode, setAuthMode] = useState<AuthMode | null>(null);

  useEffect(() => {
    getAuthMode()
      .then(setAuthMode)
      .catch(() => setAuthMode({ mode: "single", setup_complete: false }));
  }, []);

  const isSingleUser = authMode?.mode === "single";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isSingleUser) {
        await loginSimple(password);
      } else if (isSignUp) {
        await register(email, password);
      } else {
        await login(email, password);
      }
      router.push("/chat");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  if (!authMode) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glow">initializing...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="text-[#aaa] text-xs tracking-wide">
            Private · Fast · Yours
          </div>
        </div>

        <div className="glass rounded-2xl p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            {!isSingleUser && (
              <div>
                <label className="block text-[11px] text-[#aaa] mb-1.5 ml-1">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-black/40 border border-white/[0.08] text-[#f0f0f0] px-4 py-2.5 text-sm font-mono rounded-xl focus:outline-none focus:border-[#00ff41]/50 transition-all"
                  placeholder="user@example.com"
                  required
                  autoFocus
                />
              </div>
            )}

            <div>
              <label className="block text-[11px] text-[#aaa] mb-1.5 ml-1">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-black/40 border border-white/[0.08] text-[#f0f0f0] px-4 py-2.5 text-sm font-mono rounded-xl focus:outline-none focus:border-[#00ff41]/50 transition-all"
                placeholder={
                  isSingleUser && !authMode.setup_complete
                    ? "Choose a password"
                    : "••••••••"
                }
                required
                minLength={6}
                autoFocus={isSingleUser}
              />
            </div>

            {error && (
              <div className="text-[#ff3333] text-xs rounded-xl bg-[#ff3333]/[0.06] px-4 py-2.5">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-[#00ff41]/[0.15] text-[#00ff41] py-2.5 text-sm font-mono rounded-xl hover:bg-[#00ff41]/[0.22] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading
                ? "Connecting..."
                : isSingleUser
                ? authMode.setup_complete
                  ? "Sign in"
                  : "Set password & continue"
                : isSignUp
                ? "Create account"
                : "Sign in"}
            </button>
          </form>
        </div>

        {!isSingleUser && (
          <div className="mt-5 text-center">
            <button
              onClick={() => {
                setIsSignUp(!isSignUp);
                setError("");
              }}
              className="text-xs text-[#aaa] hover:text-[#00ff41] transition-colors"
            >
              {isSignUp
                ? "Already have an account? Sign in"
                : "Need an account? Sign up"}
            </button>
          </div>
        )}

        <div className="mt-8 text-center text-[10px] text-[#888]">
          v0.1.0 · All data stays local
        </div>
      </div>
    </div>
  );
}
