/**
 * CertiProof API client
 * All calls go to http://localhost:8000
 */

const BASE = "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

export type Verdict = "GENUINE" | "SUSPICIOUS" | "FAKE";

export interface FieldScore {
  value: string;
  confidence: number; // 0-100
}

export interface VerificationResult {
  id: string;
  verdict: Verdict;
  trust_score: number;
  forgery_score: number;
  field_confidence: number;
  nlp_anomaly_score: number;
  institution_match: boolean;
  institution_name: string | null;
  field_scores: Record<string, FieldScore>;
  nlp_reasoning: string;
  issues: string[];
  heatmap_url: string | null;
  report_url: string | null;
  processing_time_s: number;
  created_at: string;
  filename?: string; // injected client-side from history
}

export interface HistoryItem {
  id: string;
  filename: string;
  verdict: Verdict;
  trust_score: number;
  institution_name: string | null;
  created_at: string;
}

export interface AuthUser {
  id: string;
  email: string;
  full_name: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

// ── Token storage ─────────────────────────────────────────────────────────────

export const getToken = (): string | null => localStorage.getItem("cv_token");
export const setToken = (t: string) => localStorage.setItem("cv_token", t);
export const clearToken = () => localStorage.removeItem("cv_token");
export const getUser = (): AuthUser | null => {
  const raw = localStorage.getItem("cv_user");
  return raw ? JSON.parse(raw) : null;
};
export const setUser = (u: AuthUser) => localStorage.setItem("cv_user", JSON.stringify(u));
export const clearUser = () => localStorage.removeItem("cv_user");

// ── Helpers ───────────────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options: RequestInit = {},
  auth = false
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (auth) {
    const token = getToken();
    if (!token) throw new Error("Not authenticated");
    headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function login(email: string, password: string): Promise<AuthResponse> {
  return request<AuthResponse>("/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export async function register(
  email: string,
  password: string,
  full_name: string
): Promise<AuthResponse> {
  return request<AuthResponse>("/api/v1/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name }),
  });
}

// ── Verification ──────────────────────────────────────────────────────────────

export async function verifyCertificate(file: File): Promise<VerificationResult> {
  const form = new FormData();
  form.append("file", file);

  const token = getToken();
  if (!token) throw new Error("Not authenticated");

  // 3-minute timeout for large files on CPU inference
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3 * 60 * 1000);

  try {
    const res = await fetch(`${BASE}/api/v1/verify`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ?? `HTTP ${res.status}`);
    }
    return res.json() as Promise<VerificationResult>;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Analysis timed out after 3 minutes. Try a smaller file.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export async function getVerification(id: string): Promise<VerificationResult> {
  return request<VerificationResult>(`/api/v1/verify/${id}`, {}, true);
}

export async function getHistory(limit = 50): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/api/v1/history?limit=${limit}`, {}, true);
}

export function heatmapUrl(id: string): string {
  return `${BASE}/api/v1/heatmap/${id}?token=${getToken()}`;
}

/** Fetch heatmap as a blob URL so the token stays in the Authorization header */
export async function fetchHeatmapBlob(id: string): Promise<string | null> {
  const token = getToken();
  if (!token) return null;
  try {
    const res = await fetch(`${BASE}/api/v1/heatmap/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return null;
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────

/**
 * Stream a chat response from the backend (SSE).
 * Calls onToken for each streamed word, onDone when complete, onError on failure.
 */
export async function streamChat(
  message: string,
  verificationId: string | null,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (msg: string) => void,
): Promise<void> {
  const token = getToken();
  if (!token) { onError("Not authenticated"); return; }

  try {
    const res = await fetch(`${BASE}/api/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type":  "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify({ message, verification_id: verificationId }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      onError(body.detail ?? `HTTP ${res.status}`);
      return;
    }

    const reader  = res.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      for (const line of text.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") { onDone(); return; }
        try {
          const parsed = JSON.parse(payload);
          if (parsed.token) onToken(parsed.token);
        } catch { /* skip malformed chunks */ }
      }
    }
    onDone();
  } catch (e) {
    onError(e instanceof Error ? e.message : "Chat failed");
  }
}

// ── Send report email via Gmail SMTP ─────────────────────────────────────────
export async function sendReportEmail(verificationId: string): Promise<void> {
  return request<void>(`/api/v1/send-report/${verificationId}`, { method: "POST" }, true);
}
