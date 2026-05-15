import { useEffect, useRef, useState, CSSProperties, DragEvent } from "react";
import { jsPDF } from "jspdf";
import QRCode from "qrcode";
import {
  login, register, verifyCertificate, getHistory, getVerification,
  getToken, setToken, clearToken, getUser, setUser, clearUser,
  fetchHeatmapBlob, sendReportEmail,
  type Verdict, type VerificationResult, type HistoryItem, type AuthUser,
} from "@/lib/api";
import Chatbot from "@/components/Chatbot";

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  bg:        "#05080f",
  bgGrad:    "radial-gradient(ellipse 80% 60% at 50% -10%, #0d1f3c 0%, #05080f 70%)",
  card:      "rgba(14,20,40,0.85)",
  cardSolid: "#0e1428",
  glass:     "rgba(255,255,255,0.04)",
  border:    "rgba(59,123,248,0.15)",
  borderHi:  "rgba(59,123,248,0.35)",
  text:      "#dce8ff",
  muted:     "#4a5a8a",
  blue:      "#3b7bf8",
  blueGlow:  "rgba(59,123,248,0.25)",
  green:     "#10b981",
  greenBg:   "rgba(16,185,129,0.1)",
  greenGlow: "rgba(16,185,129,0.2)",
  amber:     "#f59e0b",
  amberBg:   "rgba(245,158,11,0.1)",
  red:       "#ef4444",
  redBg:     "rgba(239,68,68,0.1)",
  purple:    "#a855f7",
  cyan:      "#06b6d4",
};
const FONT = "'Space Grotesk', 'Inter', system-ui, sans-serif";
const MONO = "'JetBrains Mono', 'Courier New', monospace";

const STEPS = [
  "Image preprocessing & ELA",
  "Forgery detection (EfficientNet-B4)",
  "OCR & field extraction",
  "Institution database lookup",
  "Trust score fusion & verdict",
  "Saving result",
];

// ── Helpers ───────────────────────────────────────────────────────────────────
const verdictColor = (v: Verdict) =>
  v === "GENUINE"    ? { fg: C.green,  bg: C.greenBg,  glow: C.greenGlow } :
  v === "SUSPICIOUS" ? { fg: C.amber,  bg: C.amberBg,  glow: "rgba(245,158,11,0.2)" } :
                       { fg: C.red,    bg: C.redBg,    glow: "rgba(239,68,68,0.2)" };

const confColor = (c: number) => c > 80 ? C.green : c > 60 ? C.amber : C.red;

// ── Shared styles ─────────────────────────────────────────────────────────────
const card = (extra: CSSProperties = {}): CSSProperties => ({
  background: C.card,
  border: `1px solid ${C.border}`,
  borderRadius: 12,
  backdropFilter: "blur(12px)",
  WebkitBackdropFilter: "blur(12px)",
  ...extra,
});

const btnPrimary = (extra: CSSProperties = {}): CSSProperties => ({
  padding: "11px 24px",
  background: `linear-gradient(135deg, #3b7bf8, #6366f1)`,
  color: "#fff", border: "none", borderRadius: 8,
  fontFamily: FONT, fontSize: 12, letterSpacing: 1,
  cursor: "pointer", fontWeight: 700,
  boxShadow: `0 4px 20px ${C.blueGlow}`,
  transition: "all 0.2s",
  ...extra,
});

const btnGhost = (extra: CSSProperties = {}): CSSProperties => ({
  padding: "11px 24px",
  background: C.glass, color: C.text,
  border: `1px solid ${C.borderHi}`,
  fontFamily: FONT, fontSize: 12, letterSpacing: 1,
  cursor: "pointer", borderRadius: 8, transition: "all 0.2s",
  backdropFilter: "blur(8px)",
  ...extra,
});

const Label = ({ children }: { children: React.ReactNode }) => (
  <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 7, fontFamily: MONO, textTransform: "uppercase" }}>
    {children}
  </div>
);

const inputStyle = (focused: boolean): CSSProperties => ({
  width: "100%", padding: "12px 16px",
  background: focused ? "rgba(59,123,248,0.06)" : "rgba(255,255,255,0.03)",
  border: `1px solid ${focused ? C.blue : C.border}`,
  borderRadius: 8, color: C.text, fontFamily: FONT, fontSize: 13,
  outline: "none", transition: "all 0.2s", boxSizing: "border-box",
  boxShadow: focused ? `0 0 0 3px ${C.blueGlow}` : "none",
});

// ── App ───────────────────────────────────────────────────────────────────────
export default function Index() {
  const [authed, setAuthed]         = useState(!!getToken());
  const [user, setUserState]        = useState<AuthUser | null>(getUser());
  const [tab, setTab]               = useState<"upload" | "history">("upload");
  const [result, setResult]         = useState<VerificationResult | null>(null);
  const [uploading, setUploading]   = useState(false);
  const [uploadStep, setUploadStep] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleLogin = (token: string, u: AuthUser) => {
    setToken(token); setUser(u); setUserState(u); setAuthed(true);
  };

  const handleLogout = () => {
    clearToken(); clearUser(); setUserState(null);
    setAuthed(false); setResult(null); setTab("upload");
  };

  const runAnalysis = async (file: File) => {
    setUploading(true); setUploadStep(0); setResult(null); setUploadError(null);
    let step = 0;
    const ticker = setInterval(() => {
      step = Math.min(step + 1, STEPS.length - 1);
      setUploadStep(step);
    }, 900);
    try {
      const res = await verifyCertificate(file);
      clearInterval(ticker);
      setUploadStep(STEPS.length);
      await new Promise(r => setTimeout(r, 400));
      setResult({ ...res, filename: file.name });
    } catch (err) {
      clearInterval(ticker);
      setUploadError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setUploading(false);
    }
  };

  if (!authed) return <LoginPage onAuth={handleLogin} />;

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: C.bg, backgroundImage: C.bgGrad, color: C.text, fontFamily: FONT }}>
      <style>{`
        @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.85)} }
        @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        .nav-btn:hover { background: rgba(59,123,248,0.08) !important; }
        .row-hover:hover { background: rgba(59,123,248,0.05) !important; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 28px rgba(59,123,248,0.4) !important; }
        .btn-ghost:hover { background: rgba(59,123,248,0.08) !important; border-color: ${C.blue} !important; }
        .upload-zone:hover { border-color: ${C.blue} !important; background: rgba(59,123,248,0.04) !important; }
      `}</style>
      <Sidebar tab={tab} setTab={t => { setTab(t); setResult(null); }} email={user?.email ?? ""} onLogout={handleLogout} />
      <main style={{ flex: 1, overflow: "auto", padding: "36px 40px" }}>
        {tab === "upload" && (
          uploading ? <LoadingScreen step={uploadStep} />
          : result   ? <ResultPage result={result} onReset={() => setResult(null)} />
          :             <UploadPage onFile={runAnalysis} error={uploadError} />
        )}
        {tab === "history" && (
          <HistoryPage onOpen={async (id) => {
            try { const r = await getVerification(id); setTab("upload"); setResult(r); } catch {}
          }} />
        )}
      </main>
      <Chatbot verificationId={result?.id ?? null} />
    </div>
  );
}

// ── Login ─────────────────────────────────────────────────────────────────────
function LoginPage({ onAuth }: { onAuth: (token: string, user: AuthUser) => void }) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail]     = useState("");
  const [password, setPassword] = useState("");
  const [name, setName]       = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [focus, setFocus]     = useState<string | null>(null);

  const submit = async () => {
    setError(null); setLoading(true);
    try {
      const res = isRegister ? await register(email, password, name) : await login(email, password);
      onAuth(res.access_token, res.user);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Authentication failed");
    } finally { setLoading(false); }
  };

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, backgroundImage: C.bgGrad,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
    }}>
      {/* Decorative blobs */}
      <div style={{ position: "fixed", top: "15%", left: "10%", width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle, rgba(59,123,248,0.08) 0%, transparent 70%)", pointerEvents: "none" }} />
      <div style={{ position: "fixed", bottom: "10%", right: "8%", width: 300, height: 300, borderRadius: "50%", background: "radial-gradient(circle, rgba(168,85,247,0.07) 0%, transparent 70%)", pointerEvents: "none" }} />

      <div style={{ ...card(), width: 440, padding: "44px 40px", animation: "fadeIn 0.4s ease" }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <div style={{
            width: 64, height: 64, borderRadius: 16, margin: "0 auto 16px",
            background: "linear-gradient(135deg, rgba(59,123,248,0.2), rgba(168,85,247,0.2))",
            border: `1px solid ${C.borderHi}`,
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28,
          }}>🛡</div>
          <h1 style={{ fontSize: 22, letterSpacing: 5, margin: "0 0 6px", fontFamily: MONO, fontWeight: 700, background: "linear-gradient(135deg, #dce8ff, #3b7bf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            CERTIPROOF
          </h1>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 3, fontFamily: MONO }}>FORENSIC CERTIFICATE ANALYSIS</div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: 4, marginBottom: 28, border: `1px solid ${C.border}` }}>
          {(["LOGIN", "REGISTER"] as const).map((label, i) => {
            const active = isRegister === (i === 1);
            return (
              <button key={label} onClick={() => { setIsRegister(i === 1); setError(null); }} style={{
                flex: 1, padding: "9px 0",
                background: active ? "linear-gradient(135deg, rgba(59,123,248,0.25), rgba(99,102,241,0.2))" : "transparent",
                border: active ? `1px solid ${C.borderHi}` : "1px solid transparent",
                color: active ? C.text : C.muted,
                fontFamily: FONT, fontSize: 11, letterSpacing: 1.5, cursor: "pointer", borderRadius: 6,
                transition: "all 0.2s", fontWeight: active ? 700 : 400,
              }}>{label}</button>
            );
          })}
        </div>

        {isRegister && (
          <div style={{ marginBottom: 16 }}>
            <Label>Full Name</Label>
            <input value={name} onChange={e => setName(e.target.value)}
              onFocus={() => setFocus("n")} onBlur={() => setFocus(null)}
              placeholder="Your full name" style={inputStyle(focus === "n")} />
          </div>
        )}

        <div style={{ marginBottom: 16 }}>
          <Label>Email</Label>
          <input value={email} onChange={e => setEmail(e.target.value)}
            onFocus={() => setFocus("e")} onBlur={() => setFocus(null)}
            placeholder="you@example.com" style={inputStyle(focus === "e")} />
        </div>

        <div style={{ marginBottom: 4 }}>
          <Label>Password</Label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)}
            onFocus={() => setFocus("p")} onBlur={() => setFocus(null)}
            placeholder="••••••••" onKeyDown={e => e.key === "Enter" && submit()}
            style={inputStyle(focus === "p")} />
        </div>

        {error && (
          <div style={{ marginTop: 14, padding: "11px 14px", background: C.redBg, border: `1px solid rgba(239,68,68,0.3)`, borderRadius: 8, fontSize: 12, color: "#fca5a5", display: "flex", gap: 8, alignItems: "center" }}>
            <span>⚠</span><span>{error}</span>
          </div>
        )}

        <button className="btn-primary" onClick={submit} disabled={loading} style={{
          ...btnPrimary({ width: "100%", marginTop: 22, padding: "14px 0", fontSize: 12, letterSpacing: 2 }),
          opacity: loading ? 0.7 : 1, cursor: loading ? "not-allowed" : "pointer",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        }}>
          {loading && <svg width={14} height={14} viewBox="0 0 14 14" style={{ animation: "spin 0.8s linear infinite" }}><circle cx={7} cy={7} r={5} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth={2}/><path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round"/></svg>}
          {loading ? "PLEASE WAIT..." : isRegister ? "CREATE ACCOUNT" : "SIGN IN"}
        </button>

        <div style={{ textAlign: "center", marginTop: 20, fontSize: 10, color: C.muted, letterSpacing: 1.5 }}>
          🔒 SECURED WITH JWT · TLS ENCRYPTED
        </div>
      </div>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ tab, setTab, email, onLogout }: {
  tab: "upload" | "history"; setTab: (t: "upload" | "history") => void;
  email: string; onLogout: () => void;
}) {
  const navItem = (key: "upload" | "history", icon: string, label: string) => {
    const active = tab === key;
    return (
      <button className="nav-btn" onClick={() => setTab(key)} style={{
        display: "flex", alignItems: "center", gap: 10, width: "100%",
        padding: "10px 14px", marginBottom: 4,
        background: active ? "linear-gradient(135deg, rgba(59,123,248,0.15), rgba(99,102,241,0.1))" : "transparent",
        border: `1px solid ${active ? C.borderHi : "transparent"}`,
        color: active ? C.blue : C.muted,
        fontFamily: FONT, fontSize: 12, letterSpacing: 1,
        textAlign: "left", cursor: "pointer", borderRadius: 8, transition: "all 0.2s",
        boxShadow: active ? `inset 0 0 20px ${C.blueGlow}` : "none",
      }}>
        <span style={{ fontSize: 14 }}>{icon}</span>
        <span style={{ fontWeight: active ? 600 : 400 }}>{label}</span>
        {active && <div style={{ marginLeft: "auto", width: 4, height: 4, borderRadius: "50%", background: C.blue, boxShadow: `0 0 6px ${C.blue}` }} />}
      </button>
    );
  };

  return (
    <aside style={{
      width: 220, background: "rgba(8,12,24,0.9)", borderRight: `1px solid ${C.border}`,
      padding: "24px 16px", display: "flex", flexDirection: "column",
      backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
    }}>
      <div style={{ marginBottom: 32, padding: "4px 8px" }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10, marginBottom: 12,
          background: "linear-gradient(135deg, rgba(59,123,248,0.3), rgba(168,85,247,0.2))",
          border: `1px solid ${C.borderHi}`,
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
        }}>🛡</div>
        <div style={{ fontSize: 13, letterSpacing: 3, fontWeight: 700, fontFamily: MONO, background: "linear-gradient(135deg, #dce8ff, #3b7bf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          CERTIPROOF
        </div>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 1, marginTop: 3, fontFamily: MONO }}>v2.5.0</div>
      </div>

      <div style={{ marginBottom: 8, fontSize: 9, color: C.muted, letterSpacing: 2, padding: "0 8px", fontFamily: MONO }}>NAVIGATION</div>
      <div style={{ flex: 1 }}>
        {navItem("upload", "📤", "Upload")}
        {navItem("history", "🕐", "History")}
      </div>

      <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 16, marginTop: 16 }}>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 8, fontFamily: MONO }}>SIGNED IN AS</div>
        <div style={{ fontSize: 11, color: C.text, marginBottom: 14, wordBreak: "break-all", fontFamily: MONO, padding: "8px 10px", background: C.glass, borderRadius: 6, border: `1px solid ${C.border}` }}>
          {email}
        </div>
        <button className="btn-ghost" onClick={onLogout} style={{
          ...btnGhost({ width: "100%", padding: "9px 10px", fontSize: 11, letterSpacing: 1 }),
          color: C.muted,
        }}>Sign Out</button>
      </div>
    </aside>
  );
}

// ── Upload ────────────────────────────────────────────────────────────────────
function UploadPage({ onFile, error }: { onFile: (f: File) => void; error: string | null }) {
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    onFile(files[0]);
  };

  return (
    <div style={{ maxWidth: 860, animation: "fadeIn 0.3s ease" }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: "0 0 8px", fontFamily: MONO, letterSpacing: 1, background: "linear-gradient(135deg, #dce8ff 30%, #3b7bf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Certificate Analysis
        </h1>
        <p style={{ fontSize: 13, color: C.muted, margin: 0, lineHeight: 1.6 }}>
          Upload a certificate to run forensic analysis through the full AI pipeline
        </p>
      </div>

      {/* Drop zone */}
      <div
        className="upload-zone"
        onDragOver={e => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files); }}
        style={{
          padding: "60px 32px", textAlign: "center",
          background: drag ? "rgba(59,123,248,0.08)" : C.glass,
          border: `2px dashed ${drag ? C.blue : C.border}`,
          borderRadius: 16, transition: "all 0.25s", cursor: "pointer",
          boxShadow: drag ? `0 0 40px ${C.blueGlow}` : "none",
        }}
        onClick={() => fileRef.current?.click()}
      >
        <div style={{ fontSize: 48, marginBottom: 16, filter: drag ? "drop-shadow(0 0 12px rgba(59,123,248,0.6))" : "none", transition: "filter 0.3s" }}>📄</div>
        <div style={{ fontSize: 16, color: drag ? C.blue : C.text, fontWeight: 600, marginBottom: 8, letterSpacing: 0.5 }}>
          {drag ? "Drop to analyze" : "Drag & drop your certificate"}
        </div>
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 24 }}>PDF, JPG, PNG · Max 10 MB</div>
        <input ref={fileRef} type="file" accept=".pdf,.jpg,.jpeg,.png" style={{ display: "none" }}
          onChange={e => handleFiles(e.target.files)} />
        <button className="btn-primary" onClick={e => { e.stopPropagation(); fileRef.current?.click(); }}
          style={btnPrimary({ padding: "11px 28px", fontSize: 12 })}>
          Browse Files
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 16, padding: "13px 18px", background: C.redBg, border: `1px solid rgba(239,68,68,0.3)`, borderRadius: 10, fontSize: 12, color: "#fca5a5", display: "flex", gap: 10, alignItems: "center" }}>
          <span style={{ fontSize: 16 }}>⚠</span><span>{error}</span>
        </div>
      )}

      {/* Pipeline steps */}
      <div style={{ ...card({ padding: 24, marginTop: 28 }) }}>
        <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 18, fontFamily: MONO, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: C.blue }}>◆</span> ANALYSIS PIPELINE
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {STEPS.map((s, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 12, padding: "12px 14px",
              background: "rgba(255,255,255,0.02)", borderRadius: 8,
              border: `1px solid ${C.border}`, transition: "all 0.2s",
            }}>
              <div style={{
                width: 26, height: 26, borderRadius: 6, flexShrink: 0,
                background: `linear-gradient(135deg, rgba(59,123,248,0.2), rgba(99,102,241,0.15))`,
                border: `1px solid ${C.borderHi}`,
                color: C.blue, display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontFamily: MONO, fontWeight: 700,
              }}>{i + 1}</div>
              <div style={{ fontSize: 12, color: C.text, lineHeight: 1.4 }}>{s}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Loading ───────────────────────────────────────────────────────────────────
function LoadingScreen({ step }: { step: number }) {
  const pct = Math.min(100, (step / STEPS.length) * 100);
  return (
    <div style={{ maxWidth: 680, margin: "80px auto", animation: "fadeIn 0.3s ease" }}>
      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <div style={{ fontSize: 48, marginBottom: 16, animation: "pulse-dot 2s ease infinite" }}>🔍</div>
        <h1 style={{ fontSize: 24, fontWeight: 700, margin: "0 0 8px", fontFamily: MONO, letterSpacing: 2, background: "linear-gradient(135deg, #dce8ff, #3b7bf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          ANALYZING CERTIFICATE
        </h1>
        <div style={{ fontSize: 11, color: C.muted, letterSpacing: 3, fontFamily: MONO }}>RUNNING FORENSIC PIPELINE</div>
      </div>

      {/* Progress bar */}
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 99, overflow: "hidden", marginBottom: 32, border: `1px solid ${C.border}` }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 99,
          background: "linear-gradient(90deg, #3b7bf8, #a855f7)",
          transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)",
          boxShadow: "0 0 12px rgba(59,123,248,0.6)",
        }} />
      </div>

      {/* Steps */}
      <div style={{ ...card({ padding: "8px 0" }) }}>
        {STEPS.map((s, i) => {
          const done = i < step, active = i === step;
          return (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 16, padding: "14px 24px",
              borderBottom: i < STEPS.length - 1 ? `1px solid ${C.border}` : "none",
              background: active ? "rgba(59,123,248,0.05)" : "transparent",
              transition: "background 0.3s",
            }}>
              <div style={{
                width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
                background: done ? C.green : active ? C.blue : "rgba(255,255,255,0.05)",
                border: `2px solid ${done ? C.green : active ? C.blue : C.border}`,
                boxShadow: active ? `0 0 14px ${C.blueGlow}` : done ? `0 0 8px ${C.greenGlow}` : "none",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 9, color: "#fff", transition: "all 0.4s",
              }}>
                {done ? "✓" : active ? <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#fff", animation: "pulse-dot 1s ease infinite" }} /> : ""}
              </div>
              <div style={{ flex: 1, fontSize: 13, color: done ? C.text : active ? C.blue : C.muted, fontWeight: active ? 600 : 400, transition: "color 0.3s" }}>{s}</div>
              <div style={{ fontSize: 10, color: done ? C.green : active ? C.blue : C.muted, letterSpacing: 1, fontFamily: MONO, fontWeight: 600 }}>
                {done ? "✓ DONE" : active ? "RUNNING..." : "PENDING"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Result ────────────────────────────────────────────────────────────────────
interface LocalResult {
  verdict: Verdict; score: number; forgery: number; field: number; nlp: number;
  institution_match: boolean; issues: string[]; reasoning: string;
  fields: Record<string, { value: string; confidence: number }>;
  filename: string; date: string;
}

function ResultPage({ result, onReset }: { result: VerificationResult; onReset: () => void }) {
  const vc = verdictColor(result.verdict);
  const [pdfLoading, setPdfLoading]   = useState(false);
  const [pdfError, setPdfError]       = useState<string | null>(null);
  const [heatmapErr, setHeatmapErr]   = useState(false);
  const [heatmapBlob, setHeatmapBlob] = useState<string | null>(null);
  const [emailSending, setEmailSending] = useState(false);
  const [emailStatus, setEmailStatus]   = useState<"idle" | "sent" | "error">("idle");

  useEffect(() => {
    if (!result.heatmap_url) return;
    fetchHeatmapBlob(result.id).then(url => {
      if (url) setHeatmapBlob(url); else setHeatmapErr(true);
    });
    return () => { if (heatmapBlob) URL.revokeObjectURL(heatmapBlob); };
  }, [result.id]);

  const pdfResult: LocalResult = {
    verdict: result.verdict, score: result.trust_score,
    forgery: result.forgery_score, field: result.field_confidence, nlp: result.nlp_anomaly_score,
    institution_match: result.institution_match, issues: result.issues, reasoning: result.nlp_reasoning,
    fields: Object.fromEntries(Object.entries(result.field_scores).map(([k, v]) => [k, { value: v.value, confidence: v.confidence }])),
    filename: result.filename ?? result.id, date: result.created_at,
  };

  const handleDownloadPdf = async () => {
    setPdfError(null); setPdfLoading(true);
    await new Promise(r => setTimeout(r, 50));
    try { await generatePdfReport(pdfResult); }
    catch (err) { setPdfError(err instanceof Error ? err.message : String(err)); }
    finally { setPdfLoading(false); }
  };

  const handleSendEmail = async () => {
    setEmailSending(true); setEmailStatus("idle");
    try {
      await sendReportEmail(result.id);
      setEmailStatus("sent");
      setTimeout(() => setEmailStatus("idle"), 4000);
    } catch {
      setEmailStatus("error");
      setTimeout(() => setEmailStatus("idle"), 4000);
    } finally {
      setEmailSending(false);
    }
  };

  return (
    <div style={{ maxWidth: 1100, animation: "fadeIn 0.3s ease" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 8 }}>
            <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, fontFamily: MONO, letterSpacing: 1, background: "linear-gradient(135deg, #dce8ff, #3b7bf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Analysis Report
            </h1>
            <span style={{
              padding: "5px 16px", borderRadius: 999, fontSize: 11, fontWeight: 700, letterSpacing: 2,
              background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg,
              boxShadow: `0 0 16px ${vc.glow}`, fontFamily: MONO,
            }}>{result.verdict}</span>
          </div>
          <div style={{ fontSize: 11, color: C.muted, letterSpacing: 0.5, fontFamily: MONO }}>
            {result.filename ?? result.id} · {result.created_at} · {result.processing_time_s}s
          </div>
        </div>
        <button className="btn-ghost" onClick={onReset} style={btnGhost()}>← New Upload</button>
      </div>

      {/* Top row: score + metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20, marginBottom: 20 }}>
        <ScoreCard result={pdfResult} vc={vc} />
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <MetricsRow result={pdfResult} />
          <ContributionChart result={pdfResult} />
        </div>
      </div>

      {/* Anomalies */}
      {result.issues.length > 0 && (
        <div style={{ ...card({ padding: 20, marginBottom: 20, background: C.redBg, borderColor: "rgba(239,68,68,0.3)" }) }}>
          <div style={{ fontSize: 11, color: C.red, letterSpacing: 2, marginBottom: 12, fontWeight: 700, fontFamily: MONO, display: "flex", alignItems: "center", gap: 8 }}>
            <span>⚠</span> DETECTED ANOMALIES ({result.issues.length})
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {result.issues.map((iss, i) => (
              <div key={i} style={{ fontSize: 12, color: "#fca5a5", padding: "8px 12px", background: "rgba(239,68,68,0.08)", borderRadius: 6, border: "1px solid rgba(239,68,68,0.15)", display: "flex", gap: 8 }}>
                <span style={{ color: C.red, flexShrink: 0 }}>→</span>{iss}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bottom row: fields + reasoning + heatmap */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
        <FieldsCard result={pdfResult} />
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <ReasoningCard result={pdfResult} />
          <HeatmapCard imgSrc={heatmapBlob} hasError={heatmapErr} onError={() => setHeatmapErr(true)} />
        </div>
      </div>

      {/* PDF error */}
      {pdfError && (
        <div style={{ ...card({ padding: "14px 18px", marginBottom: 16, background: C.redBg, borderColor: "rgba(239,68,68,0.3)" }), display: "flex", alignItems: "flex-start", gap: 12 }}>
          <span style={{ color: C.red, fontSize: 16, flexShrink: 0 }}>⚠</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: C.red, letterSpacing: 2, fontWeight: 700, fontFamily: MONO, marginBottom: 4 }}>PDF GENERATION FAILED</div>
            <div style={{ fontSize: 12, color: "#fca5a5", lineHeight: 1.6 }}>{pdfError}</div>
          </div>
          <button onClick={() => setPdfError(null)} style={{ background: "transparent", border: "none", color: C.red, fontSize: 18, cursor: "pointer", padding: 0, lineHeight: 1 }}>✕</button>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <button className="btn-primary" onClick={handleDownloadPdf} disabled={pdfLoading}
          style={{ ...btnPrimary({ display: "flex", alignItems: "center", gap: 8 }), opacity: pdfLoading ? 0.7 : 1, cursor: pdfLoading ? "not-allowed" : "pointer" }}>
          {pdfLoading ? <><svg width={13} height={13} viewBox="0 0 14 14" style={{ animation: "spin 0.8s linear infinite" }}><circle cx={7} cy={7} r={5} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth={2}/><path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round"/></svg>Generating...</> : "📥 Download PDF Report"}
        </button>
        <button className="btn-ghost" style={btnGhost()} onClick={() => navigator.clipboard?.writeText(
          `CertiProof Report — ${result.filename ?? result.id}\nVerdict: ${result.verdict} (Score ${result.trust_score}/100)`
        )}>🔗 Share Result</button>
        <button className="btn-ghost" onClick={onReset} style={btnGhost()}>Verify Another</button>
        <button
          className="btn-ghost"
          onClick={handleSendEmail}
          disabled={emailSending}
          style={{
            ...btnGhost({
              display: "flex", alignItems: "center", gap: 8,
              opacity: emailSending ? 0.7 : 1,
              cursor: emailSending ? "not-allowed" : "pointer",
              borderColor: emailStatus === "sent" ? C.green : emailStatus === "error" ? C.red : undefined,
              color: emailStatus === "sent" ? C.green : emailStatus === "error" ? C.red : undefined,
            }),
          }}
        >
          {emailSending ? (
            <><svg width={13} height={13} viewBox="0 0 14 14" style={{ animation: "spin 0.8s linear infinite" }}><circle cx={7} cy={7} r={5} fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={2}/><path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"/></svg>Sending...</>
          ) : emailStatus === "sent" ? "✓ Email Sent!"
          : emailStatus === "error" ? "✕ Failed"
          : "📧 Send to Email"}
        </button>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────
function ScoreCard({ result, vc }: { result: LocalResult; vc: { fg: string; bg: string; glow: string } }) {
  const r = 52, circ = 2 * Math.PI * r;
  const offset = circ - (result.score / 100) * circ;
  return (
    <div style={{ ...card({ padding: 28, textAlign: "center" }) }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 3, marginBottom: 20, fontFamily: MONO }}>TRUST SCORE</div>
      <div style={{ position: "relative", width: 150, height: 150, margin: "0 auto" }}>
        <svg width={150} height={150} viewBox="0 0 130 130" style={{ transform: "rotate(-90deg)" }}>
          <circle cx={65} cy={65} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={10} />
          <circle cx={65} cy={65} r={r} fill="none" stroke={vc.fg} strokeWidth={10}
            strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)", filter: `drop-shadow(0 0 8px ${vc.glow})` }} />
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontSize: 42, color: vc.fg, fontWeight: 800, fontFamily: MONO, lineHeight: 1 }}>{result.score}</div>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1, fontFamily: MONO }}>/100</div>
        </div>
      </div>
      <div style={{ marginTop: 20 }}>
        <span style={{
          display: "inline-block", padding: "8px 24px", borderRadius: 999,
          background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg,
          fontSize: 11, letterSpacing: 3, fontWeight: 700, fontFamily: MONO,
          boxShadow: `0 0 20px ${vc.glow}`,
        }}>{result.verdict}</span>
      </div>
      <div style={{ marginTop: 20, padding: "12px 14px", background: "rgba(255,255,255,0.03)", borderRadius: 8, border: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 6, fontFamily: MONO }}>INSTITUTION</div>
        <div style={{ fontSize: 12, color: result.institution_match ? C.green : C.red, letterSpacing: 1, fontFamily: MONO, fontWeight: 600 }}>
          {result.institution_match ? "✓ VERIFIED" : "✕ NOT FOUND"}
        </div>
      </div>
    </div>
  );
}

function MetricsRow({ result }: { result: LocalResult }) {
  const cards = [
    { label: "FORGERY", sublabel: "ELA Score", value: result.forgery, color: C.blue, icon: "🔬" },
    { label: "FIELDS",  sublabel: "OCR Conf.",  value: result.field,   color: C.purple, icon: "📋" },
    { label: "NLP",     sublabel: "Reasoning",  value: result.nlp,     color: C.cyan, icon: "🧠" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
      {cards.map(c => (
        <div key={c.label} style={{ ...card({ padding: "18px 16px" }) }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, fontFamily: MONO }}>{c.label}</div>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{c.sublabel}</div>
            </div>
            <span style={{ fontSize: 18 }}>{c.icon}</span>
          </div>
          <div style={{ fontSize: 32, color: c.color, fontWeight: 800, fontFamily: MONO, lineHeight: 1 }}>
            {Math.round(c.value * 100)}<span style={{ fontSize: 16, opacity: 0.6 }}>%</span>
          </div>
          <div style={{ marginTop: 10, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ width: `${Math.round(c.value * 100)}%`, height: "100%", background: c.color, borderRadius: 99, transition: "width 1s ease", boxShadow: `0 0 6px ${c.color}` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function ContributionChart({ result }: { result: LocalResult }) {
  const bars = [
    { label: "Forgery Detection", weight: 45, color: C.blue },
    { label: "Field Confidence",  weight: 35, color: C.purple },
    { label: "NLP Reasoning",     weight: 20, color: C.cyan },
  ];
  return (
    <div style={{ ...card({ padding: 20 }) }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 16, fontFamily: MONO, display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: C.blue }}>◆</span> SCORE CONTRIBUTION
      </div>
      {bars.map(b => (
        <div key={b.label} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: C.text, width: 130, flexShrink: 0 }}>{b.label}</div>
          <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.05)", borderRadius: 99, overflow: "hidden", border: `1px solid ${C.border}` }}>
            <div style={{ width: `${b.weight}%`, height: "100%", background: `linear-gradient(90deg, ${b.color}, ${b.color}88)`, borderRadius: 99, transition: "width 1s ease" }} />
          </div>
          <div style={{ fontSize: 11, color: b.color, width: 36, textAlign: "right", fontFamily: MONO, fontWeight: 700 }}>{b.weight}%</div>
        </div>
      ))}
    </div>
  );
}

function FieldsCard({ result }: { result: LocalResult }) {
  return (
    <div style={{ ...card({ padding: 22 }) }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 16, fontFamily: MONO, display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: C.blue }}>◆</span> EXTRACTED FIELDS
      </div>
      {Object.entries(result.fields).map(([k, v], i, arr) => (
        <div key={k} style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "10px 0", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none",
        }}>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1.5, fontFamily: MONO, width: 110, flexShrink: 0 }}>{k}</div>
          <div style={{ flex: 1, fontSize: 12, color: C.text, fontFamily: MONO, fontWeight: 500, textAlign: "right", paddingRight: 12 }}>{v.value}</div>
          <div style={{
            fontSize: 10, color: confColor(v.confidence), letterSpacing: 1,
            fontFamily: MONO, fontWeight: 700, minWidth: 38, textAlign: "right",
            padding: "2px 8px", borderRadius: 4,
            background: `${confColor(v.confidence)}18`,
          }}>{v.confidence}%</div>
        </div>
      ))}
    </div>
  );
}

function ReasoningCard({ result }: { result: LocalResult }) {
  return (
    <div style={{ ...card({ padding: 22 }) }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 14, fontFamily: MONO, display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: C.purple }}>◆</span> AI REASONING
      </div>
      <div style={{ fontSize: 13, color: C.text, lineHeight: 1.8, opacity: 0.9 }}>{result.reasoning}</div>
    </div>
  );
}

function HeatmapCard({ imgSrc, hasError, onError }: { imgSrc: string | null; hasError: boolean; onError: () => void }) {
  return (
    <div style={{ ...card({ padding: 22 }) }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 14, fontFamily: MONO, display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: C.cyan }}>◆</span> GRADCAM HEATMAP
      </div>
      {imgSrc && !hasError ? (
        <img src={imgSrc} alt="GradCAM heatmap" onError={onError}
          style={{ width: "100%", borderRadius: 8, border: `1px solid ${C.border}` }} />
      ) : (
        <div style={{
          height: 110, borderRadius: 8, border: `1px dashed ${C.border}`,
          background: "linear-gradient(135deg, rgba(10,21,48,0.8), rgba(26,10,48,0.8), rgba(42,10,10,0.8))",
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8,
        }}>
          <span style={{ fontSize: 24, opacity: 0.4 }}>🗺</span>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, fontFamily: MONO }}>
            {hasError ? "HEATMAP UNAVAILABLE" : "NO HEATMAP"}
          </div>
        </div>
      )}
    </div>
  );
}

// ── History ───────────────────────────────────────────────────────────────────
function HistoryPage({ onOpen }: { onOpen: (id: string) => void }) {
  const [items, setItems]     = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    getHistory().then(setItems).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ maxWidth: 900, animation: "fadeIn 0.3s ease" }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: "0 0 8px", fontFamily: MONO, letterSpacing: 1, background: "linear-gradient(135deg, #dce8ff 30%, #3b7bf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Analysis History
        </h1>
        <p style={{ fontSize: 13, color: C.muted, margin: 0 }}>
          {loading ? "Loading..." : error ? error : `${items.length} verification${items.length !== 1 ? "s" : ""}`}
        </p>
      </div>

      <div style={{ ...card({ overflow: "hidden", padding: 0 }) }}>
        {/* Table header */}
        <div style={{
          display: "grid", gridTemplateColumns: "2fr 1fr 80px 1.4fr",
          padding: "12px 20px", background: "rgba(255,255,255,0.03)",
          borderBottom: `1px solid ${C.border}`,
          fontSize: 9, color: C.muted, letterSpacing: 2, fontFamily: MONO,
        }}>
          <div>FILENAME</div><div>VERDICT</div><div>SCORE</div><div>DATE</div>
        </div>

        {items.length === 0 && !loading && (
          <div style={{ padding: "48px 20px", textAlign: "center" }}>
            <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>📂</div>
            <div style={{ fontSize: 13, color: C.muted }}>No verifications yet</div>
          </div>
        )}

        {items.map((h, i) => {
          const vc = verdictColor(h.verdict);
          return (
            <div key={h.id} className="row-hover" onClick={() => onOpen(h.id)} style={{
              display: "grid", gridTemplateColumns: "2fr 1fr 80px 1.4fr",
              padding: "16px 20px", borderBottom: i < items.length - 1 ? `1px solid ${C.border}` : "none",
              cursor: "pointer", alignItems: "center", transition: "background 0.2s",
            }}>
              <div>
                <div style={{ fontSize: 13, color: C.text, fontFamily: MONO, fontWeight: 500, marginBottom: 3 }}>{h.filename}</div>
                <div style={{ fontSize: 11, color: C.muted }}>{h.institution_name ?? "—"}</div>
              </div>
              <div>
                <span style={{
                  fontSize: 9, padding: "4px 12px", borderRadius: 999,
                  background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg,
                  letterSpacing: 1.5, fontWeight: 700, fontFamily: MONO,
                  boxShadow: `0 0 8px ${vc.glow}`,
                }}>{h.verdict}</span>
              </div>
              <div style={{ fontSize: 18, color: vc.fg, fontWeight: 800, fontFamily: MONO }}>{h.trust_score}</div>
              <div style={{ fontSize: 11, color: C.muted, fontFamily: MONO }}>{h.created_at}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── PDF Report ────────────────────────────────────────────────────────────────
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function drawCard(doc: jsPDF, x: number, y: number, w: number, h: number) {
  doc.setFillColor(...hexToRgb("#0e1428"));
  doc.setDrawColor(...hexToRgb("#1a2340"));
  doc.setLineWidth(0.5);
  doc.roundedRect(x, y, w, h, 4, 4, "FD");
}

async function generatePdfReport(result: LocalResult) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const H = doc.internal.pageSize.getHeight();
  const M = 40;
  const FOOTER_Y = H - 50;
  const CONT_H = 56;
  const vc = verdictColor(result.verdict);
  const [tr, tg, tb] = hexToRgb("#dce8ff");
  const [mr, mg, mb] = hexToRgb("#4a5a8a");
  const [vr, vg, vb] = hexToRgb(vc.fg);
  const [vbr, vbg, vbb] = hexToRgb(vc.bg.replace("rgba(", "").replace(")", "").split(",").slice(0,3).map(Number).reduce((acc, v, i) => { acc[i] = v; return acc; }, [] as number[]).map(v => Math.round(v)).join("").padStart(6, "0") || "#0e1428");
  const [br, bg, bb] = hexToRgb("#05080f");
  const [cr, cg, cb] = hexToRgb("#0e1428");
  const [bdr, bdg, bdb] = hexToRgb("#243060");

  const fillBg = () => { doc.setFillColor(br, bg, bb); doc.rect(0, 0, W, H, "F"); };

  const newPage = (): number => {
    doc.addPage(); fillBg();
    doc.setFillColor(cr, cg, cb); doc.rect(0, 0, W, CONT_H, "F");
    doc.setDrawColor(bdr, bdg, bdb); doc.setLineWidth(0.5); doc.line(0, CONT_H, W, CONT_H);
    doc.setFont("courier", "bold"); doc.setFontSize(11); doc.setTextColor(tr, tg, tb); doc.text("CERTIPROOF", M, 24);
    doc.setFont("courier", "normal"); doc.setFontSize(8); doc.setTextColor(mr, mg, mb);
    doc.text("FORENSIC CERTIFICATE ANALYSIS REPORT (CONTINUED)", M, 38);
    doc.setFillColor(vbr, vbg, vbb); doc.setDrawColor(vr, vg, vb); doc.setLineWidth(0.8);
    doc.roundedRect(W - M - 90, 14, 90, 24, 12, 12, "FD");
    doc.setTextColor(vr, vg, vb); doc.setFont("courier", "bold"); doc.setFontSize(10);
    doc.text(result.verdict, W - M - 45, 30, { align: "center" });
    return CONT_H + 16;
  };

  const ensureSpace = (y: number, needed: number) => y + needed > FOOTER_Y ? newPage() : y;

  fillBg();
  doc.setFillColor(cr, cg, cb); doc.rect(0, 0, W, 70, "F");
  doc.setDrawColor(bdr, bdg, bdb); doc.setLineWidth(0.5); doc.line(0, 70, W, 70);
  doc.setFont("courier", "bold"); doc.setFontSize(18); doc.setTextColor(tr, tg, tb); doc.text("CERTIPROOF", M, 32);
  doc.setFont("courier", "normal"); doc.setFontSize(8); doc.setTextColor(mr, mg, mb);
  doc.text("FORENSIC CERTIFICATE ANALYSIS REPORT", M, 48);
  doc.text(`GENERATED ${new Date().toISOString().slice(0, 19).replace("T", " ")} UTC`, M, 60);
  doc.setFillColor(vbr, vbg, vbb); doc.setDrawColor(vr, vg, vb); doc.setLineWidth(1);
  doc.roundedRect(W - M - 110, 22, 110, 30, 15, 15, "FD");
  doc.setTextColor(vr, vg, vb); doc.setFont("courier", "bold"); doc.setFontSize(13);
  doc.text(result.verdict, W - M - 55, 41, { align: "center" });

  let y = 100;
  drawCard(doc, M, y, W - 2 * M, 60);
  doc.setFont("courier", "normal"); doc.setFontSize(8); doc.setTextColor(mr, mg, mb);
  doc.text("FILENAME", M + 14, y + 16); doc.text("ANALYZED", M + 14, y + 38);
  doc.setTextColor(tr, tg, tb); doc.setFontSize(11);
  doc.text(result.filename, M + 90, y + 16); doc.text(result.date, M + 90, y + 38);
  y += 80;

  y = ensureSpace(y, 160);
  drawCard(doc, M, y, 180, 140);
  doc.setFont("courier", "normal"); doc.setFontSize(8); doc.setTextColor(mr, mg, mb);
  doc.text("TRUST SCORE", M + 90, y + 18, { align: "center" });
  doc.setFont("courier", "bold"); doc.setFontSize(48); doc.setTextColor(vr, vg, vb);
  doc.text(String(result.score), M + 90, y + 70, { align: "center" });
  doc.setFontSize(9); doc.text("/ 100", M + 90, y + 88, { align: "center" });
  doc.setFont("courier", "normal"); doc.setFontSize(8); doc.setTextColor(mr, mg, mb);
  doc.text("INSTITUTION MATCH", M + 90, y + 110, { align: "center" });
  doc.setTextColor(...(result.institution_match ? hexToRgb("#10b981") : hexToRgb("#ef4444")));
  doc.setFont("courier", "bold"); doc.setFontSize(10);
  doc.text(result.institution_match ? "VERIFIED" : "NOT FOUND", M + 90, y + 124, { align: "center" });

  const metrics = [
    { label: "FORGERY SCORE", v: result.forgery, color: "#3b7bf8", weight: "45%" },
    { label: "FIELD CONFIDENCE", v: result.field, color: "#a855f7", weight: "35%" },
    { label: "NLP REASONING", v: result.nlp, color: "#06b6d4", weight: "20%" },
  ];
  const mxStart = M + 200, mw = (W - 2 * M - 200 - 20) / 3;
  metrics.forEach((m, i) => {
    const x = mxStart + i * (mw + 10);
    drawCard(doc, x, y, mw, 65);
    doc.setFont("courier", "normal"); doc.setFontSize(7); doc.setTextColor(mr, mg, mb); doc.text(m.label, x + 10, y + 16);
    doc.setTextColor(...hexToRgb(m.color)); doc.setFont("courier", "bold"); doc.setFontSize(22);
    doc.text(`${Math.round(m.v * 100)}%`, x + 10, y + 44);
    doc.setFontSize(7); doc.setTextColor(mr, mg, mb); doc.setFont("courier", "normal"); doc.text(`WEIGHT ${m.weight}`, x + 10, y + 58);
  });
  y += 160;

  if (result.issues.length > 0) {
    const rowH = 14, headerH = 30, blockH = headerH + result.issues.length * rowH + 8;
    y = ensureSpace(y, blockH);
    doc.setFillColor(42, 10, 10); doc.setDrawColor(...hexToRgb("#ef4444")); doc.setLineWidth(0.8);
    doc.roundedRect(M, y, W - 2 * M, blockH, 4, 4, "FD");
    doc.setFont("courier", "bold"); doc.setFontSize(9); doc.setTextColor(...hexToRgb("#ef4444"));
    doc.text(`⚠ DETECTED ANOMALIES (${result.issues.length})`, M + 12, y + 16);
    doc.setFont("courier", "normal"); doc.setFontSize(9); doc.setTextColor(252, 165, 165);
    result.issues.forEach((iss, i) => doc.text(`→ ${iss}`, M + 14, y + headerH + i * rowH));
    y += blockH + 16;
  }

  const fEntries = Object.entries(result.fields);
  const ROW_H = 20, SEC_H = 28;
  y = ensureSpace(y, SEC_H + ROW_H);
  doc.setFont("courier", "bold"); doc.setFontSize(9); doc.setTextColor(mr, mg, mb); doc.text("EXTRACTED FIELDS", M + 12, y + 14);
  doc.setDrawColor(bdr, bdg, bdb); doc.setLineWidth(0.5);
  doc.line(M, y, W - M, y); doc.line(M, y + SEC_H, W - M, y + SEC_H);
  y += SEC_H;
  fEntries.forEach(([k, v], i) => {
    y = ensureSpace(y, ROW_H + 4);
    if (i % 2 === 0) { doc.setFillColor(cr, cg, cb); doc.rect(M, y, W - 2 * M, ROW_H, "F"); }
    doc.setTextColor(mr, mg, mb); doc.setFont("courier", "normal"); doc.setFontSize(8); doc.text(k, M + 12, y + 13);
    doc.setTextColor(tr, tg, tb); doc.setFontSize(10); doc.text(v.value, M + 180, y + 13);
    const cc = v.confidence > 80 ? "#10b981" : v.confidence > 60 ? "#f59e0b" : "#ef4444";
    doc.setTextColor(...hexToRgb(cc)); doc.setFont("courier", "bold"); doc.setFontSize(9);
    doc.text(`${v.confidence}%`, W - M - 14, y + 13, { align: "right" });
    doc.setDrawColor(cr, cg, cb); doc.setLineWidth(0.3); doc.line(M, y + ROW_H, W - M, y + ROW_H);
    y += ROW_H;
  });
  doc.setDrawColor(bdr, bdg, bdb); doc.setLineWidth(0.5); doc.line(M, y, W - M, y); y += 20;

  const LINE_H = 14;
  const reasonLines: string[] = doc.splitTextToSize(result.reasoning, W - 2 * M - 24);
  y = ensureSpace(y, SEC_H + LINE_H);
  doc.setFont("courier", "bold"); doc.setFontSize(9); doc.setTextColor(mr, mg, mb); doc.text("AI REASONING", M + 12, y + 14);
  doc.setDrawColor(bdr, bdg, bdb); doc.setLineWidth(0.5);
  doc.line(M, y, W - M, y); doc.line(M, y + SEC_H, W - M, y + SEC_H);
  y += SEC_H + 4;
  doc.setFont("courier", "normal"); doc.setFontSize(10); doc.setTextColor(tr, tg, tb);
  reasonLines.forEach(line => { y = ensureSpace(y, LINE_H + 4); doc.text(line, M + 12, y + LINE_H); y += LINE_H; });
  y += 12;

  const qrUrl = `https://certiproof.io/verify/${result.filename.replace(/\.[^.]+$/, "")}`;
  const qrDataUrl = await QRCode.toDataURL(qrUrl, { width: 64, margin: 0, color: { dark: "#dce8ff", light: "#05080f" } });
  const pages = doc.getNumberOfPages();
  const QR = 32, QR_X = W - M - QR, QR_Y = H - 30 - QR / 2;
  for (let p = 1; p <= pages; p++) {
    doc.setPage(p);
    doc.setDrawColor(cr, cg, cb); doc.setLineWidth(0.3); doc.line(M, H - 30, W - M, H - 30);
    doc.setFont("courier", "normal"); doc.setFontSize(7); doc.setTextColor(mr, mg, mb);
    doc.text("CERTIPROOF · CONFIDENTIAL · JWT-AUTH · TLS ENCRYPTED", M, H - 18);
    doc.text(`PAGE ${p} / ${pages}`, W - M - QR - 10, H - 18, { align: "right" });
    doc.addImage(qrDataUrl, "PNG", QR_X, QR_Y, QR, QR);
  }
  doc.save(`certiproof_${result.filename.replace(/\.[^.]+$/, "")}_report.pdf`);
}
