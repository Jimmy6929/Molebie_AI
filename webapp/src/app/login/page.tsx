"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const supabase = createClient();

    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        // For local Supabase, auto-confirm is usually on
        const { error: loginError } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (loginError) throw loginError;
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) throw error;
      }
      router.push("/chat");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen flex items-center justify-center px-4 bg-[#050505]">
      <div className="w-full max-w-sm glass rounded-2xl p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="text-2xl mb-3 opacity-30">⌘</div>
          <div className="text-sm text-[#00ff41] glow font-bold tracking-wider mb-1">LOCAL AI</div>
          <div className="text-[10px] text-[#555]">private · fast · yours</div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-[11px] text-[#555] mb-1.5">
              email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl text-[#00ff41] px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-[#00ff41]/30 focus:bg-white/[0.05] transition-all"
              placeholder="user@example.com"
              required
              autoFocus
            />
          </div>

          <div>
            <label className="block text-[11px] text-[#555] mb-1.5">
              password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl text-[#00ff41] px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-[#00ff41]/30 focus:bg-white/[0.05] transition-all"
              placeholder="••••••••"
              required
              minLength={6}
            />
          </div>

          {error && (
            <div className="text-[#ff3333] text-[11px] border border-[#ff3333]/20 bg-[#ff3333]/5 px-4 py-2.5 rounded-xl">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#00ff41]/10 border border-[#00ff41]/20 text-[#00ff41] py-2.5 text-sm font-mono rounded-xl hover:bg-[#00ff41]/20 hover:border-[#00ff41]/30 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {loading
              ? "connecting…"
              : isSignUp
              ? "create account"
              : "login"}
          </button>
        </form>

        <div className="mt-5 text-center">
          <button
            onClick={() => {
              setIsSignUp(!isSignUp);
              setError("");
            }}
            className="text-[11px] text-[#555] hover:text-[#00ff41] transition-all"
          >
            {isSignUp
              ? "already have an account? login"
              : "need an account? sign up"}
          </button>
        </div>

        <div className="mt-6 text-center text-[10px] text-[#333]">
          v0.1.0 · all data stays local
        </div>
      </div>
    </div>
  );
}
