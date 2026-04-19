// Resolve the gateway base URL for API calls.
//
// Priority:
//   1. An explicit NEXT_PUBLIC_GATEWAY_URL set to a non-default value
//      (distributed setups where the gateway lives on a different host).
//   2. In a browser, derive from the current origin's hostname so access via
//      LAN / Tailscale / mDNS / localhost all route to the gateway running on
//      whichever machine the user just opened.
//   3. Fall back to localhost for SSR / build time.

const DEFAULT_ENV_VALUES = new Set<string>([
  "",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
]);

const GATEWAY_PORT = 8000;

export function resolveGatewayUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "";

  if (envUrl && !DEFAULT_ENV_VALUES.has(envUrl)) {
    return envUrl;
  }

  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host && host !== "localhost" && host !== "127.0.0.1") {
      return `${window.location.protocol}//${host}:${GATEWAY_PORT}`;
    }
  }

  return envUrl || `http://localhost:${GATEWAY_PORT}`;
}

export const GATEWAY_URL = resolveGatewayUrl();
