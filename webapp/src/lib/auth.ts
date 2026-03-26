/**
 * Gateway-managed authentication client.
 *
 * Replaces Supabase Auth. Stores JWT in localStorage.
 * Supports single-user (password only) and multi-user (email + password) modes.
 */

const GATEWAY_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

const TOKEN_KEY = "molebie_token";

export interface AuthUser {
  id: string;
  email: string;
}

export interface AuthResponse {
  token: string;
  user: AuthUser;
}

export interface AuthMode {
  mode: "single" | "multi";
  setup_complete: boolean;
}

/** Get the stored JWT token, or null if not logged in. */
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

/** Store the JWT token after login. */
function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

/** Clear the stored token (logout). */
export function logout(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/** Check if user is logged in (has a token). */
export function isLoggedIn(): boolean {
  return !!getToken();
}

/** Fetch the auth mode (single/multi) from the gateway. */
export async function getAuthMode(): Promise<AuthMode> {
  const res = await fetch(`${GATEWAY_URL}/auth/mode`);
  if (!res.ok) throw new Error("Failed to get auth mode");
  return res.json();
}

/** Login with email + password (multi-user mode). */
export async function login(
  email: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${GATEWAY_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(data.detail || "Login failed");
  }
  const data: AuthResponse = await res.json();
  setToken(data.token);
  return data;
}

/** Register with email + password (multi-user mode). */
export async function register(
  email: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${GATEWAY_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const data = await res
      .json()
      .catch(() => ({ detail: "Registration failed" }));
    throw new Error(data.detail || "Registration failed");
  }
  const data: AuthResponse = await res.json();
  setToken(data.token);
  return data;
}

/** Login with password only (single-user mode). */
export async function loginSimple(password: string): Promise<AuthResponse> {
  const res = await fetch(`${GATEWAY_URL}/auth/login-simple`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(data.detail || "Login failed");
  }
  const data: AuthResponse = await res.json();
  setToken(data.token);
  return data;
}
