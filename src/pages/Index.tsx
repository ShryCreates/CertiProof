import { useEffect, useRef, useState, CSSProperties, DragEvent } from "react";
import { jsPDF } from "jspdf";
import QRCode from "qrcode";

// ============ DESIGN TOKENS ============
const C = {
  bg: "#080c18",
  card: "#0e1428",
  border: "#1a2340",
  borderHi: "#243060",
  text: "#c9d4f0",
  muted: "#5a6a9a",
  blue: "#3b7bf8",
  green: "#10b981",
  greenBg: "#0d2e22",
  amber: "#f59e0b",
  amberBg: "#2d1f06",
  red: "#ef4444",
  redBg: "#2a0a0a",
  purple: "#a855f7",
  cyan: "#06b6d4",
};
const FONT = "'Space Grotesk', 'Inter', system-ui, sans-serif";
const MONO = "'JetBrains Mono', 'Courier New', monospace";

// ============ MOCK DATA ============
type Verdict = "GENUINE" | "SUSPICIOUS" | "FAKE";
interface FieldVal { value: string; confidence: number; }
interface Result {
  verdict: Verdict;
  score: number;
  forgery: number;
  field: number;
  nlp: number;
  institution_match: boolean;
  issues: string[];
  reasoning: string;
  fields: Record<string, FieldVal>;
  filename: string;
  date: string;
}

const RESULTS: Record<Verdict, Result> = {
  GENUINE: {
    verdict: "GENUINE", score: 87, forgery: 0.91, field: 0.88, nlp: 0.79,
    institution_match: true, issues: [],
    reasoning: "All extracted fields are internally consistent. The institution seal, signature DPI, and font kerning match the verified template for IIT Bombay. ELA analysis shows uniform compression with no tampered regions. Grade and roll number formats align with the institution's known schema.",
    fields: {
      "STUDENT NAME": { value: "Arjun Mehta", confidence: 96 },
      "INSTITUTION": { value: "IIT Bombay", confidence: 98 },
      "DEGREE": { value: "B.Tech", confidence: 95 },
      "DISCIPLINE": { value: "Computer Science", confidence: 93 },
      "ISSUE DATE": { value: "12 June 2023", confidence: 91 },
      "GRADE": { value: "8.7 CGPA", confidence: 94 },
      "ROLL NUMBER": { value: "180050042", confidence: 97 },
    },
    filename: "iitb_btech_2023.pdf", date: "2026-05-04 14:22",
  },
  SUSPICIOUS: {
    verdict: "SUSPICIOUS", score: 58, forgery: 0.62, field: 0.55, nlp: 0.51,
    institution_match: true,
    issues: ["Grade region shows ELA anomalies", "Roll number format mismatch", "Logo position deviation"],
    reasoning: "Moderate compression artifacts detected near grade field, indicating possible localized edits. Roll number format deviates from Delhi University's standard schema (expected 11 digits, found 10). Institution logo position is shifted ~12px from canonical reference. Recommend manual review.",
    fields: {
      "STUDENT NAME": { value: "Priya Sharma", confidence: 84 },
      "INSTITUTION": { value: "Delhi University", confidence: 89 },
      "DEGREE": { value: "B.A. (Hons)", confidence: 78 },
      "DISCIPLINE": { value: "Economics", confidence: 72 },
      "ISSUE DATE": { value: "08 July 2022", confidence: 65 },
      "GRADE": { value: "7.4 CGPA", confidence: 48 },
      "ROLL NUMBER": { value: "1820041", confidence: 42 },
    },
    filename: "du_ba_2022.jpg", date: "2026-05-03 11:08",
  },
  FAKE: {
    verdict: "FAKE", score: 24, forgery: 0.18, field: 0.31, nlp: 0.22,
    institution_match: false,
    issues: ["Institution not in verified DB", "Implausible grade value (9.9 CGPA)", "Multiple ELA tamper regions", "Signature DPI mismatch"],
    reasoning: "High ELA anomaly scores across multiple regions including signature, seal, and grade. Institution 'Royal Indian Tech University' is not present in the verified institution database. Grade value of 9.9 CGPA is statistically improbable for the claimed program. Signature DPI (72) inconsistent with document DPI (300).",
    fields: {
      "STUDENT NAME": { value: "Rohit Kumar", confidence: 71 },
      "INSTITUTION": { value: "Royal Indian Tech University", confidence: 28 },
      "DEGREE": { value: "M.Tech", confidence: 55 },
      "DISCIPLINE": { value: "AI & ML", confidence: 49 },
      "ISSUE DATE": { value: "30 Feb 2024", confidence: 22 },
      "GRADE": { value: "9.9 CGPA", confidence: 18 },
      "ROLL NUMBER": { value: "RIT99999", confidence: 35 },
    },
    filename: "ritu_mtech_2024.pdf", date: "2026-05-02 19:45",
  },
};

const HISTORY: { v: Verdict; filename: string; institution: string; date: string }[] = [
  { v: "GENUINE", filename: "iitb_btech_2023.pdf", institution: "IIT Bombay", date: "2026-05-04 14:22" },
  { v: "SUSPICIOUS", filename: "du_ba_2022.jpg", institution: "Delhi University", date: "2026-05-03 11:08" },
  { v: "FAKE", filename: "ritu_mtech_2024.pdf", institution: "Royal Indian Tech University", date: "2026-05-02 19:45" },
  { v: "GENUINE", filename: "nit_trichy_be.pdf", institution: "NIT Trichy", date: "2026-05-01 09:30" },
  { v: "SUSPICIOUS", filename: "anonymous_cert.png", institution: "VTU Belagavi", date: "2026-04-30 16:55" },
];

const STEPS = [
  "Image preprocessing & ELA",
  "Forgery detection (EfficientNet-B4)",
  "OCR & field extraction (LayoutLMv3)",
  "Institution database lookup",
  "LLM reasoning (Mistral-7B)",
  "Trust score fusion & verdict",
];

// ============ HELPERS ============
const verdictColor = (v: Verdict) =>
  v === "GENUINE" ? { fg: C.green, bg: C.greenBg } :
  v === "SUSPICIOUS" ? { fg: C.amber, bg: C.amberBg } :
  { fg: C.red, bg: C.redBg };

const confColor = (c: number) => c > 80 ? C.green : c > 60 ? C.amber : C.red;

// ============ APP ============
export default function Index() {
  const [page, setPage] = useState<"login" | "app">("login");
  const [isRegister, setIsRegister] = useState(false);
  const [tab, setTab] = useState<"upload" | "history">("upload");
  const [result, setResult] = useState<Result | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStep, setUploadStep] = useState(0);
  const [email, setEmail] = useState("analyst@certvalidator.io");

  const startAnalysis = (verdict: Verdict) => {
    setUploading(true);
    setUploadStep(0);
    setResult(null);
    let s = 0;
    const tick = () => {
      s += 1;
      setUploadStep(s);
      if (s < STEPS.length) setTimeout(tick, 600);
      else setTimeout(() => { setUploading(false); setResult(RESULTS[verdict]); }, 400);
    };
    setTimeout(tick, 400);
  };

  if (page === "login") {
    return <LoginPage
      isRegister={isRegister} setIsRegister={setIsRegister}
      email={email} setEmail={setEmail}
      onLogin={() => setPage("app")}
    />;
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: C.bg, color: C.text, fontFamily: FONT }}>
      <Sidebar tab={tab} setTab={(t) => { setTab(t); }} email={email} onLogout={() => { setPage("login"); setResult(null); setTab("upload"); }} />
      <main style={{ flex: 1, overflow: "auto", padding: 32 }}>
        {tab === "upload" && (
          uploading ? <LoadingScreen step={uploadStep} />
          : result ? <ResultPage result={result} onReset={() => setResult(null)} />
          : <UploadPage onTrigger={startAnalysis} />
        )}
        {tab === "history" && (
          <HistoryPage onOpen={(v) => { setTab("upload"); setResult(RESULTS[v]); }} />
        )}
      </main>
    </div>
  );
}

// ============ LOGIN ============
function LoginPage({ isRegister, setIsRegister, email, setEmail, onLogin }: {
  isRegister: boolean; setIsRegister: (b: boolean) => void;
  email: string; setEmail: (s: string) => void; onLogin: () => void;
}) {
  const [pw, setPw] = useState("••••••••");
  const [focus, setFocus] = useState<string | null>(null);
  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: FONT, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div style={{ width: 420, background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 36 }}>
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{ fontSize: 42 }}>🛡</div>
          <h1 style={{ fontSize: 24, letterSpacing: 4, margin: "10px 0 4px", fontFamily: MONO, fontWeight: 700 }}>CERTVALIDATOR</h1>
          <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, fontFamily: MONO }}>FORENSIC CERTIFICATE ANALYSIS</div>
        </div>
        <div style={{ display: "flex", borderBottom: `1px solid ${C.border}`, marginBottom: 24 }}>
          {[["LOGIN", false], ["REGISTER", true]].map(([label, reg]) => (
            <button key={label as string}
              onClick={() => setIsRegister(reg as boolean)}
              style={{
                flex: 1, padding: "12px 0", background: "transparent", border: "none",
                color: isRegister === reg ? C.blue : C.muted,
                borderBottom: isRegister === reg ? `2px solid ${C.blue}` : "2px solid transparent",
                fontFamily: FONT, fontSize: 12, letterSpacing: 1, cursor: "pointer", transition: "all 0.2s",
              }}>{label as string}</button>
          ))}
        </div>
        <Label>EMAIL</Label>
        <input value={email} onChange={(e) => setEmail(e.target.value)}
          onFocus={() => setFocus("e")} onBlur={() => setFocus(null)}
          style={inputStyle(focus === "e")} />
        <div style={{ height: 14 }} />
        <Label>PASSWORD</Label>
        <input type="password" value={pw} onChange={(e) => setPw(e.target.value)}
          onFocus={() => setFocus("p")} onBlur={() => setFocus(null)}
          style={inputStyle(focus === "p")} />
        <button onClick={onLogin} style={{
          width: "100%", marginTop: 24, padding: "14px 0", background: C.blue, color: "#fff",
          border: "none", borderRadius: 6, fontFamily: FONT, fontSize: 13, letterSpacing: 2,
          cursor: "pointer", fontWeight: "bold", transition: "all 0.2s",
        }}>{isRegister ? "REGISTER" : "LOGIN"}</button>
        <div style={{ textAlign: "center", marginTop: 22, fontSize: 10, color: C.muted, letterSpacing: 1 }}>
          SECURED WITH JWT AUTH · TLS ENCRYPTED
        </div>
      </div>
    </div>
  );
}

const inputStyle = (focused: boolean): CSSProperties => ({
  width: "100%", padding: "12px 14px", background: C.bg,
  border: `1px solid ${focused ? C.blue : C.border}`, borderRadius: 6,
  color: C.text, fontFamily: FONT, fontSize: 13, outline: "none", transition: "all 0.2s",
  boxSizing: "border-box",
});

const Label = ({ children }: { children: React.ReactNode }) =>
  <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1.5, marginBottom: 6, fontFamily: MONO }}>{children}</div>;

// ============ SIDEBAR ============
function Sidebar({ tab, setTab, email, onLogout }: {
  tab: "upload" | "history"; setTab: (t: "upload" | "history") => void;
  email: string; onLogout: () => void;
}) {
  const navItem = (key: "upload" | "history", icon: string, label: string) => {
    const active = tab === key;
    return (
      <button onClick={() => setTab(key)} style={{
        display: "flex", alignItems: "center", gap: 10, width: "100%",
        padding: "11px 14px", marginBottom: 4,
        background: active ? "#0c1838" : "transparent",
        border: `1px solid ${active ? C.borderHi : "transparent"}`,
        color: active ? C.blue : C.text,
        fontFamily: FONT, fontSize: 12, letterSpacing: 1,
        textAlign: "left", cursor: "pointer", borderRadius: 6, transition: "all 0.2s",
      }}>
        <span>{icon}</span><span>{label}</span>
      </button>
    );
  };
  return (
    <aside style={{ width: 200, background: C.card, borderRight: `1px solid ${C.border}`, padding: 18, display: "flex", flexDirection: "column" }}>
      <div style={{ marginBottom: 28, padding: "4px 6px" }}>
        <div style={{ fontSize: 22 }}>🛡</div>
        <div style={{ fontSize: 14, letterSpacing: 3, marginTop: 6, fontWeight: 700, fontFamily: MONO }}>CERTVALIDATOR</div>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 1, marginTop: 4, fontFamily: MONO }}>v2.4.1</div>
      </div>
      <div style={{ flex: 1 }}>
        {navItem("upload", "📤", "UPLOAD")}
        {navItem("history", "🕐", "HISTORY")}
      </div>
      <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 14, marginTop: 14 }}>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 6, fontFamily: MONO }}>SIGNED IN</div>
        <div style={{ fontSize: 11, color: C.text, marginBottom: 12, wordBreak: "break-all", fontFamily: MONO }}>{email}</div>
        <button onClick={onLogout} style={{
          width: "100%", padding: "8px 10px", background: "transparent",
          border: `1px solid ${C.border}`, color: C.muted, fontFamily: FONT,
          fontSize: 11, letterSpacing: 1, cursor: "pointer", borderRadius: 6, transition: "all 0.2s",
        }}>LOGOUT</button>
      </div>
    </aside>
  );
}

// ============ UPLOAD ============
function UploadPage({ onTrigger }: { onTrigger: (v: Verdict) => void }) {
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault(); setDrag(false);
    onTrigger("SUSPICIOUS");
  };

  const demoBtn = (label: string, color: string, bg: string, v: Verdict) => (
    <button onClick={() => onTrigger(v)} style={{
      flex: 1, padding: "14px 16px", background: bg, border: `1px solid ${color}`,
      color, fontFamily: FONT, fontSize: 11, letterSpacing: 1, cursor: "pointer",
      borderRadius: 6, transition: "all 0.2s", fontWeight: "bold",
    }}>DEMO: {label}</button>
  );

  return (
    <div>
      <h1 style={{ fontSize: 26, letterSpacing: 1, margin: 0, fontWeight: 700, fontFamily: MONO }}>CERTIFICATE ANALYSIS</h1>
      <div style={{ fontSize: 12, color: C.muted, marginTop: 6, letterSpacing: 0.5 }}>
        Upload a certificate to run forensic analysis through the full AI pipeline
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        style={{
          marginTop: 24, padding: "56px 32px", textAlign: "center",
          background: drag ? "#0c1838" : C.card,
          border: `2px dashed ${drag ? C.blue : C.border}`,
          borderRadius: 10, transition: "all 0.2s",
        }}>
        <div style={{ fontSize: 42, marginBottom: 14 }}>📤</div>
        <div style={{ fontSize: 14, color: C.text, letterSpacing: 1, marginBottom: 6 }}>
          DRAG & DROP CERTIFICATE
        </div>
        <div style={{ fontSize: 11, color: C.muted, marginBottom: 18 }}>
          PDF, JPG, PNG · Max 10MB
        </div>
        <input ref={fileRef} type="file" style={{ display: "none" }} onChange={() => onTrigger("GENUINE")} />
        <button onClick={() => fileRef.current?.click()} style={{
          padding: "10px 22px", background: "transparent", border: `1px solid ${C.blue}`,
          color: C.blue, fontFamily: FONT, fontSize: 11, letterSpacing: 1, cursor: "pointer",
          borderRadius: 6, transition: "all 0.2s",
        }}>BROWSE FILES</button>
      </div>

      <div style={{ display: "flex", gap: 12, marginTop: 18 }}>
        {demoBtn("GENUINE", C.green, C.greenBg, "GENUINE")}
        {demoBtn("SUSPICIOUS", C.amber, C.amberBg, "SUSPICIOUS")}
        {demoBtn("FAKE", C.red, C.redBg, "FAKE")}
      </div>

      <div style={{ marginTop: 28, background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
        <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 14, fontFamily: MONO }}>
          📋 ANALYSIS PIPELINE
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {STEPS.map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: 10, background: C.bg, borderRadius: 6, border: `1px solid ${C.border}` }}>
              <div style={{ width: 22, height: 22, borderRadius: 4, background: "#0c1838", color: C.blue, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11 }}>{i + 1}</div>
              <div style={{ fontSize: 12 }}>{s}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============ LOADING ============
function LoadingScreen({ step }: { step: number }) {
  const pct = Math.min(100, (step / STEPS.length) * 100);
  return (
    <div style={{ maxWidth: 720, margin: "60px auto" }}>
      <h1 style={{ fontSize: 26, letterSpacing: 1, margin: 0, textAlign: "center", fontWeight: 700, fontFamily: MONO }}>ANALYZING CERTIFICATE...</h1>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, textAlign: "center", marginTop: 8, fontFamily: MONO }}>
        RUNNING FORENSIC PIPELINE
      </div>
      <div style={{ marginTop: 32, height: 6, background: C.border, borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: C.blue, transition: "width 0.5s ease" }} />
      </div>
      <div style={{ marginTop: 32, background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 24 }}>
        {STEPS.map((s, i) => {
          const done = i < step;
          const active = i === step;
          const color = done ? C.green : active ? C.blue : C.muted;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 0", borderBottom: i < STEPS.length - 1 ? `1px solid ${C.border}` : "none" }}>
              <div style={{
                width: 12, height: 12, borderRadius: "50%",
                background: done ? C.green : active ? C.blue : "#1a2340",
                boxShadow: active ? `0 0 12px ${C.blue}` : "none",
                transition: "all 0.3s",
              }} />
              <div style={{ fontSize: 12, color, letterSpacing: 0.5, flex: 1 }}>{s}</div>
              <div style={{ fontSize: 10, color, letterSpacing: 1 }}>
                {done ? "✓ DONE" : active ? "RUNNING..." : "PENDING"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============ RESULT ============
function ResultPage({ result, onReset }: { result: Result; onReset: () => void }) {
  const vc = verdictColor(result.verdict);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const handleDownloadPdf = async () => {
    setPdfError(null);
    setPdfLoading(true);
    // Yield to the browser so the state update renders before the synchronous PDF work blocks the thread
    await new Promise((resolve) => setTimeout(resolve, 50));
    try {
      await generatePdfReport(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setPdfError(`PDF generation failed: ${message}`);
    } finally {
      setPdfLoading(false);
    }
  };
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: 26, letterSpacing: 1, margin: 0, fontWeight: 700, fontFamily: MONO }}>ANALYSIS REPORT</h1>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 8, letterSpacing: 1, fontFamily: MONO }}>
            {result.filename} · {result.date}
          </div>
        </div>
        <button onClick={onReset} style={btnGhost()}>← NEW UPLOAD</button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20, marginTop: 24 }}>
        <ScoreCard result={result} vc={vc} />
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <MetricsRow result={result} />
          <ContributionChart result={result} />
        </div>
      </div>

      {result.issues.length > 0 && (
        <div style={{ marginTop: 20, background: C.redBg, border: `1px solid ${C.red}`, borderRadius: 8, padding: 18 }}>
          <div style={{ fontSize: 11, color: C.red, letterSpacing: 2, marginBottom: 10, fontWeight: 700, fontFamily: MONO }}>
            ⚠ DETECTED ANOMALIES ({result.issues.length})
          </div>
          {result.issues.map((iss, i) => (
            <div key={i} style={{ fontSize: 12, color: "#ffb3b3", padding: "4px 0" }}>→ {iss}</div>
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 20 }}>
        <FieldsCard result={result} />
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <ReasoningCard result={result} />
          <HeatmapCard />
        </div>
      </div>

      {pdfError && (
        <div role="alert" style={{
          marginTop: 20, padding: "14px 18px",
          background: C.redBg, border: `1px solid ${C.red}`, borderRadius: 8,
          display: "flex", alignItems: "flex-start", gap: 12,
        }}>
          <span style={{ fontSize: 16, lineHeight: 1, flexShrink: 0 }}>⚠</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: C.red, letterSpacing: 2, fontWeight: 700, fontFamily: MONO, marginBottom: 4 }}>
              PDF GENERATION FAILED
            </div>
            <div style={{ fontSize: 12, color: "#ffb3b3", lineHeight: 1.6 }}>{pdfError}</div>
          </div>
          <button
            onClick={() => setPdfError(null)}
            aria-label="Dismiss error"
            style={{
              background: "transparent", border: "none", color: C.red,
              fontSize: 16, cursor: "pointer", lineHeight: 1, flexShrink: 0, padding: 0,
            }}
          >✕</button>
        </div>
      )}

      <div style={{ display: "flex", gap: 12, marginTop: 24, flexWrap: "wrap" }}>
        <button
          style={{ ...btnPrimary(), opacity: pdfLoading ? 0.65 : 1, cursor: pdfLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 8 }}          onClick={handleDownloadPdf}
          disabled={pdfLoading}
        >
          {pdfLoading ? (
            <>
              <svg width={14} height={14} viewBox="0 0 14 14" style={{ animation: "spin 0.8s linear infinite", flexShrink: 0 }}>
                <circle cx={7} cy={7} r={5} fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth={2} />
                <path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round" />
              </svg>
              GENERATING...
            </>
          ) : "📥 DOWNLOAD PDF REPORT"}
        </button>
        <button style={btnGhost()} onClick={() => {
          navigator.clipboard?.writeText(`CertValidator Report — ${result.filename}\nVerdict: ${result.verdict} (Score ${result.score}/100)`);
        }}>🔗 SHARE RESULT</button>
        <button onClick={onReset} style={btnGhost()}>VERIFY ANOTHER</button>
      </div>
    </div>
  );
}

const btnPrimary = (): CSSProperties => ({
  padding: "12px 22px", background: C.blue, color: "#fff", border: "none",
  fontFamily: FONT, fontSize: 11, letterSpacing: 1, cursor: "pointer",
  borderRadius: 6, fontWeight: "bold", transition: "all 0.2s",
});
const btnGhost = (): CSSProperties => ({
  padding: "12px 22px", background: "transparent", color: C.text,
  border: `1px solid ${C.borderHi}`, fontFamily: FONT, fontSize: 11,
  letterSpacing: 1, cursor: "pointer", borderRadius: 6, transition: "all 0.2s",
});

// ============ PDF REPORT ============
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

async function generatePdfReport(result: Result) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const H = doc.internal.pageSize.getHeight();
  const M = 40;
  // Usable vertical range: below the page-1 header (100pt) / below continuation header (56pt), above footer (50pt)
  const FOOTER_Y = H - 50;
  const CONT_HEADER_H = 56; // height of the slim continuation header on pages 2+
  const vc = verdictColor(result.verdict);

  // ── colour shortcuts ──────────────────────────────────────────────────────
  const [br, bg, bb] = hexToRgb(C.bg);
  const [cr, cg, cb] = hexToRgb(C.card);
  const [bdr, bdg, bdb] = hexToRgb(C.borderHi);
  const [tr, tg, tb] = hexToRgb(C.text);
  const [mr, mg, mb] = hexToRgb(C.muted);
  const [vr, vg, vb] = hexToRgb(vc.fg);
  const [vbr, vbg, vbb] = hexToRgb(vc.bg);

  // ── helpers ───────────────────────────────────────────────────────────────

  /** Fill the current page with the dark background colour. */
  const fillPageBg = () => {
    doc.setFillColor(br, bg, bb);
    doc.rect(0, 0, W, H, "F");
  };

  /**
   * Add a new page, paint its background, draw a slim continuation header,
   * and return the y cursor ready for content.
   */
  const newPage = (): number => {
    doc.addPage();
    fillPageBg();
    // slim header bar
    doc.setFillColor(cr, cg, cb);
    doc.rect(0, 0, W, CONT_HEADER_H, "F");
    doc.setDrawColor(bdr, bdg, bdb);
    doc.setLineWidth(0.5);
    doc.line(0, CONT_HEADER_H, W, CONT_HEADER_H);
    doc.setFont("courier", "bold");
    doc.setFontSize(11);
    doc.setTextColor(tr, tg, tb);
    doc.text("CERTVALIDATOR", M, 24);
    doc.setFont("courier", "normal");
    doc.setFontSize(8);
    doc.setTextColor(mr, mg, mb);
    doc.text("FORENSIC CERTIFICATE ANALYSIS REPORT (CONTINUED)", M, 38);
    // verdict pill (right side)
    doc.setFillColor(vbr, vbg, vbb);
    doc.setDrawColor(vr, vg, vb);
    doc.setLineWidth(0.8);
    doc.roundedRect(W - M - 90, 14, 90, 24, 12, 12, "FD");
    doc.setTextColor(vr, vg, vb);
    doc.setFont("courier", "bold");
    doc.setFontSize(10);
    doc.text(result.verdict, W - M - 45, 30, { align: "center" });
    return CONT_HEADER_H + 16;
  };

  /**
   * Ensure there is at least `needed` pts of space below `y`.
   * If not, starts a new page and returns the fresh y cursor.
   */
  const ensureSpace = (y: number, needed: number): number =>
    y + needed > FOOTER_Y ? newPage() : y;

  // ── PAGE 1 header ─────────────────────────────────────────────────────────
  fillPageBg();
  doc.setFillColor(cr, cg, cb);
  doc.rect(0, 0, W, 70, "F");
  doc.setDrawColor(bdr, bdg, bdb);
  doc.setLineWidth(0.5);
  doc.line(0, 70, W, 70);

  doc.setFont("courier", "bold");
  doc.setFontSize(18);
  doc.setTextColor(tr, tg, tb);
  doc.text("CERTVALIDATOR", M, 32);

  doc.setFont("courier", "normal");
  doc.setFontSize(8);
  doc.setTextColor(mr, mg, mb);
  doc.text("FORENSIC CERTIFICATE ANALYSIS REPORT", M, 48);
  doc.text(`GENERATED ${new Date().toISOString().slice(0, 19).replace("T", " ")} UTC`, M, 60);

  // verdict pill
  doc.setFillColor(vbr, vbg, vbb);
  doc.setDrawColor(vr, vg, vb);
  doc.setLineWidth(1);
  doc.roundedRect(W - M - 110, 22, 110, 30, 15, 15, "FD");
  doc.setTextColor(vr, vg, vb);
  doc.setFont("courier", "bold");
  doc.setFontSize(13);
  doc.text(result.verdict, W - M - 55, 41, { align: "center" });

  let y = 100;

  // ── File info card (fixed height, always fits on page 1) ──────────────────
  drawCard(doc, M, y, W - 2 * M, 60);
  doc.setFont("courier", "normal");
  doc.setFontSize(8);
  doc.setTextColor(mr, mg, mb);
  doc.text("FILENAME", M + 14, y + 16);
  doc.text("ANALYZED", M + 14, y + 38);
  doc.setTextColor(tr, tg, tb);
  doc.setFontSize(11);
  doc.text(result.filename, M + 90, y + 16);
  doc.text(result.date, M + 90, y + 38);
  y += 80;

  // ── Trust score + metrics row (fixed height 160pt) ────────────────────────
  y = ensureSpace(y, 160);
  drawCard(doc, M, y, 180, 140);
  doc.setFont("courier", "normal");
  doc.setFontSize(8);
  doc.setTextColor(mr, mg, mb);
  doc.text("TRUST SCORE", M + 90, y + 18, { align: "center" });
  doc.setFont("courier", "bold");
  doc.setFontSize(48);
  doc.setTextColor(vr, vg, vb);
  doc.text(String(result.score), M + 90, y + 70, { align: "center" });
  doc.setFontSize(9);
  doc.text("/ 100", M + 90, y + 88, { align: "center" });
  doc.setFont("courier", "normal");
  doc.setFontSize(8);
  doc.setTextColor(mr, mg, mb);
  doc.text("INSTITUTION MATCH", M + 90, y + 110, { align: "center" });
  doc.setTextColor(...(result.institution_match ? hexToRgb(C.green) : hexToRgb(C.red)));
  doc.setFont("courier", "bold");
  doc.setFontSize(10);
  doc.text(result.institution_match ? "VERIFIED" : "NOT FOUND", M + 90, y + 124, { align: "center" });

  const metrics = [
    { label: "FORGERY SCORE", v: result.forgery, color: C.blue, weight: "45%" },
    { label: "FIELD CONFIDENCE", v: result.field, color: C.purple, weight: "35%" },
    { label: "NLP REASONING", v: result.nlp, color: C.cyan, weight: "20%" },
  ];
  const mxStart = M + 200;
  const mw = (W - 2 * M - 200 - 20) / 3;
  metrics.forEach((m, i) => {
    const x = mxStart + i * (mw + 10);
    drawCard(doc, x, y, mw, 65);
    doc.setFont("courier", "normal");
    doc.setFontSize(7);
    doc.setTextColor(mr, mg, mb);
    doc.text(m.label, x + 10, y + 16);
    const [r, g, b] = hexToRgb(m.color);
    doc.setTextColor(r, g, b);
    doc.setFont("courier", "bold");
    doc.setFontSize(22);
    doc.text(`${Math.round(m.v * 100)}%`, x + 10, y + 44);
    doc.setFontSize(7);
    doc.setTextColor(mr, mg, mb);
    doc.setFont("courier", "normal");
    doc.text(`WEIGHT ${m.weight}`, x + 10, y + 58);
  });

  const barSectionY = y + 75;
  drawCard(doc, mxStart, barSectionY, W - mxStart - M, 65);
  doc.setFont("courier", "normal");
  doc.setFontSize(7);
  doc.setTextColor(mr, mg, mb);
  doc.text("SCORE CONTRIBUTION (WEIGHTED)", mxStart + 10, barSectionY + 14);
  const barX = mxStart + 70;
  const barMaxW = W - mxStart - M - 110;
  metrics.forEach((m, i) => {
    const by = barSectionY + 26 + i * 13;
    const bw = barMaxW * (parseInt(m.weight) / 100);
    doc.setTextColor(tr, tg, tb);
    doc.setFontSize(7);
    doc.text(m.label.split(" ")[0], mxStart + 10, by + 7);
    doc.setFillColor(...hexToRgb(C.bg));
    doc.rect(barX, by, barMaxW, 8, "F");
    doc.setFillColor(...hexToRgb(m.color));
    doc.rect(barX, by, bw, 8, "F");
    doc.setTextColor(...hexToRgb(m.color));
    doc.text(m.weight, barX + barMaxW + 6, by + 7);
  });
  y += 160;

  // ── Anomalies (each row is 14pt; guard the whole block or row-by-row) ─────
  if (result.issues.length > 0) {
    const [rr, rg, rb] = hexToRgb(C.red);
    const rowH = 14;
    const headerH = 30; // label + top padding
    const blockH = headerH + result.issues.length * rowH + 8;

    y = ensureSpace(y, blockH);

    doc.setFillColor(...hexToRgb(C.redBg));
    doc.setDrawColor(rr, rg, rb);
    doc.setLineWidth(0.8);
    doc.roundedRect(M, y, W - 2 * M, blockH, 4, 4, "FD");
    doc.setFont("courier", "bold");
    doc.setFontSize(9);
    doc.setTextColor(rr, rg, rb);
    doc.text(`⚠ DETECTED ANOMALIES (${result.issues.length})`, M + 12, y + 16);
    doc.setFont("courier", "normal");
    doc.setFontSize(9);
    doc.setTextColor(255, 179, 179);
    result.issues.forEach((iss, i) => {
      doc.text(`→ ${iss}`, M + 14, y + headerH + i * rowH);
    });
    y += blockH + 16;
  }

  // ── Extracted Fields — row-by-row with page breaks ────────────────────────
  const fEntries = Object.entries(result.fields);
  const ROW_H = 20;
  const SECTION_HEADER_H = 28;

  // Section heading — needs at least heading + 1 row
  y = ensureSpace(y, SECTION_HEADER_H + ROW_H);

  // Draw the section label (no bounding box — we'll draw rows individually)
  doc.setFont("courier", "bold");
  doc.setFontSize(9);
  doc.setTextColor(mr, mg, mb);
  doc.text("EXTRACTED FIELDS", M + 12, y + 14);

  // thin top border line
  doc.setDrawColor(...hexToRgb(C.borderHi));
  doc.setLineWidth(0.5);
  doc.line(M, y, W - M, y);
  doc.line(M, y + SECTION_HEADER_H, W - M, y + SECTION_HEADER_H);

  y += SECTION_HEADER_H;

  fEntries.forEach(([k, v], i) => {
    y = ensureSpace(y, ROW_H + 4);

    // row background (alternating subtle tint)
    if (i % 2 === 0) {
      doc.setFillColor(cr, cg, cb);
      doc.rect(M, y, W - 2 * M, ROW_H, "F");
    }

    doc.setTextColor(mr, mg, mb);
    doc.setFont("courier", "normal");
    doc.setFontSize(8);
    doc.text(k, M + 12, y + 13);

    doc.setTextColor(tr, tg, tb);
    doc.setFontSize(10);
    doc.text(v.value, M + 180, y + 13);

    doc.setTextColor(...hexToRgb(confColor(v.confidence)));
    doc.setFont("courier", "bold");
    doc.setFontSize(9);
    doc.text(`${v.confidence}%`, W - M - 14, y + 13, { align: "right" });

    // row divider
    doc.setDrawColor(...hexToRgb(C.border));
    doc.setLineWidth(0.3);
    doc.line(M, y + ROW_H, W - M, y + ROW_H);

    y += ROW_H;
  });

  // closing border
  doc.setDrawColor(...hexToRgb(C.borderHi));
  doc.setLineWidth(0.5);
  doc.line(M, y, W - M, y);
  y += 20;

  // ── LLM Reasoning — line-by-line with page breaks ─────────────────────────
  const LINE_H = 14; // pt per line at fontSize 10
  const reasonLines: string[] = doc.splitTextToSize(result.reasoning, W - 2 * M - 24);

  // Section heading — needs at least heading + 1 line
  y = ensureSpace(y, SECTION_HEADER_H + LINE_H);

  doc.setFont("courier", "bold");
  doc.setFontSize(9);
  doc.setTextColor(mr, mg, mb);
  doc.text("LLM REASONING (MISTRAL-7B)", M + 12, y + 14);
  doc.setDrawColor(...hexToRgb(C.borderHi));
  doc.setLineWidth(0.5);
  doc.line(M, y, W - M, y);
  doc.line(M, y + SECTION_HEADER_H, W - M, y + SECTION_HEADER_H);
  y += SECTION_HEADER_H + 4;

  doc.setFont("courier", "normal");
  doc.setFontSize(10);
  doc.setTextColor(tr, tg, tb);

  reasonLines.forEach((line) => {
    y = ensureSpace(y, LINE_H + 4);
    doc.text(line, M + 12, y + LINE_H);
    y += LINE_H;
  });

  y += 12; // trailing gap

  // ── Footer on every page ──────────────────────────────────────────────────
  // Generate QR code linking to this certificate result
  const verificationUrl = `https://certvalidator.io/verify/${result.filename.replace(/\.[^.]+$/, "")}`;
  const qrDataUrl = await QRCode.toDataURL(verificationUrl, {
    width: 64,
    margin: 0,
    color: { dark: C.text, light: C.bg },
  });

  const pages = doc.getNumberOfPages();
  const QR_SIZE = 32;
  const QR_X = W - M - QR_SIZE;
  const QR_Y = H - 30 - QR_SIZE / 2;

  for (let p = 1; p <= pages; p++) {
    doc.setPage(p);
    doc.setDrawColor(...hexToRgb(C.border));
    doc.setLineWidth(0.3);
    doc.line(M, H - 30, W - M, H - 30);
    doc.setFont("courier", "normal");
    doc.setFontSize(7);
    doc.setTextColor(mr, mg, mb);
    doc.text("CERTVALIDATOR · CONFIDENTIAL · JWT-AUTH · TLS ENCRYPTED", M, H - 18);
    doc.text(`PAGE ${p} / ${pages}`, W - M - QR_SIZE - 10, H - 18, { align: "right" });
    
    // QR code in bottom-right corner
    doc.addImage(qrDataUrl, "PNG", QR_X, QR_Y, QR_SIZE, QR_SIZE);
  }

  doc.save(`certvalidator_${result.filename.replace(/\.[^.]+$/, "")}_report.pdf`);
}

function drawCard(doc: jsPDF, x: number, y: number, w: number, h: number) {
  const [cr, cg, cb] = hexToRgb(C.card);
  const [br, bg, bb] = hexToRgb(C.border);
  doc.setFillColor(cr, cg, cb);
  doc.setDrawColor(br, bg, bb);
  doc.setLineWidth(0.5);
  doc.roundedRect(x, y, w, h, 4, 4, "FD");
}

// ============ SCORE RING ============
function ScoreCard({ result, vc }: { result: Result; vc: { fg: string; bg: string } }) {
  const r = 46, c = 2 * Math.PI * r;
  const offset = c - (result.score / 100) * c;
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 24, textAlign: "center" }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 3, marginBottom: 18, fontFamily: MONO }}>VERDICT</div>
      <div style={{ position: "relative", width: 140, height: 140, margin: "0 auto" }}>
        <svg width={140} height={140} viewBox="0 0 120 120" style={{ transform: "rotate(-90deg)" }}>
          <circle cx={60} cy={60} r={r} fill="none" stroke={C.border} strokeWidth={8} />
          <circle cx={60} cy={60} r={r} fill="none" stroke={vc.fg} strokeWidth={8}
            strokeLinecap="round" strokeDasharray={c} strokeDashoffset={offset}
            style={{ transition: "stroke-dasharray 1s ease, stroke-dashoffset 1s ease" }} />
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
          <div style={{ fontSize: 38, color: vc.fg, fontWeight: 700, letterSpacing: 1, fontFamily: MONO }}>{result.score}</div>
        </div>
      </div>
      <div style={{ fontSize: 9, color: C.muted, letterSpacing: 3, marginTop: 8, fontFamily: MONO }}>TRUST SCORE</div>
      <div style={{ marginTop: 18 }}>
        <span style={{
          display: "inline-block", padding: "9px 22px", borderRadius: 999,
          background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg,
          fontSize: 12, letterSpacing: 3, fontWeight: 700, fontFamily: MONO,
        }}>{result.verdict}</span>
      </div>
      <div style={{ marginTop: 22, padding: 12, background: C.bg, borderRadius: 6, border: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 6, fontFamily: MONO }}>INSTITUTION MATCH</div>
        <div style={{ fontSize: 13, color: result.institution_match ? C.green : C.red, letterSpacing: 1.5, fontFamily: MONO, fontWeight: 600 }}>
          {result.institution_match ? "✓ VERIFIED" : "✕ NOT FOUND"}
        </div>
      </div>
    </div>
  );
}

function MetricsRow({ result }: { result: Result }) {
  const cards = [
    { label: "FORGERY SCORE", value: result.forgery, color: C.blue },
    { label: "FIELD CONFIDENCE", value: result.field, color: C.purple },
    { label: "NLP REASONING", value: result.nlp, color: C.cyan },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
      {cards.map((c) => (
        <div key={c.label} style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 2, marginBottom: 10, fontFamily: MONO }}>{c.label}</div>
          <div style={{ fontSize: 30, color: c.color, fontWeight: 700, fontFamily: MONO, letterSpacing: -0.5 }}>{Math.round(c.value * 100)}<span style={{ fontSize: 16, opacity: 0.7 }}>%</span></div>
        </div>
      ))}
    </div>
  );
}

function ContributionChart({ result }: { result: Result }) {
  const bars = [
    { label: "FORGERY", weight: 45, value: result.forgery, color: C.blue },
    { label: "FIELD", weight: 35, value: result.field, color: C.purple },
    { label: "NLP", weight: 20, value: result.nlp, color: C.cyan },
  ];
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 16, fontFamily: MONO }}>SCORE CONTRIBUTION (WEIGHTED)</div>
      {bars.map((b) => (
        <div key={b.label} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: C.text, letterSpacing: 1.5, width: 80, fontFamily: MONO, fontWeight: 600 }}>{b.label}</div>
          <div style={{ flex: 1, height: 18, background: C.bg, borderRadius: 4, overflow: "hidden", border: `1px solid ${C.border}` }}>
            <div style={{
              width: `${b.weight}%`, height: "100%",
              background: b.color, transition: "width 0.8s ease",
            }} />
          </div>
          <div style={{ fontSize: 12, color: b.color, width: 48, textAlign: "right", letterSpacing: 1, fontFamily: MONO, fontWeight: 700 }}>{b.weight}%</div>
        </div>
      ))}
    </div>
  );
}

function FieldsCard({ result }: { result: Result }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 14, fontFamily: MONO }}>EXTRACTED FIELDS</div>
      {Object.entries(result.fields).map(([k, v], i, arr) => (
        <div key={k} style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "10px 0", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none",
        }}>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1.5, fontFamily: MONO }}>{k}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ fontSize: 13, color: C.text, fontFamily: MONO, fontWeight: 500 }}>{v.value}</div>
            <div style={{ fontSize: 10, color: confColor(v.confidence), letterSpacing: 1, minWidth: 38, textAlign: "right", fontFamily: MONO, fontWeight: 600 }}>
              {v.confidence}%
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ReasoningCard({ result }: { result: Result }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 14, fontFamily: MONO }}>🧠 LLM REASONING</div>
      <div style={{ fontSize: 13, color: C.text, lineHeight: 1.75 }}>{result.reasoning}</div>
    </div>
  );
}

function HeatmapCard() {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 12, fontFamily: MONO }}>🗺 GRADCAM HEATMAP</div>
      <div style={{
        height: 120, borderRadius: 6, border: `1px dashed ${C.border}`,
        background: "linear-gradient(135deg, #0a1530 0%, #1a0a30 50%, #2a0a0a 100%)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 12, fontFamily: MONO,
      }}>HEATMAP PREVIEW</div>
      <button style={{ ...btnGhost(), width: "100%" }}>VIEW OVERLAY</button>
    </div>
  );
}

// ============ HISTORY ============
function HistoryPage({ onOpen }: { onOpen: (v: Verdict) => void }) {
  return (
    <div>
      <h1 style={{ fontSize: 26, letterSpacing: 1, margin: 0, fontWeight: 700, fontFamily: MONO }}>ANALYSIS HISTORY</h1>
      <div style={{ fontSize: 12, color: C.muted, marginTop: 6, letterSpacing: 0.5 }}>
        {HISTORY.length} records · last 30 days
      </div>
      <div style={{ marginTop: 24, background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, overflow: "hidden" }}>
        <div style={{
          display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1.4fr",
          padding: "14px 20px", background: C.bg, borderBottom: `1px solid ${C.border}`,
          fontSize: 10, color: C.muted, letterSpacing: 2, fontFamily: MONO,
        }}>
          <div>FILENAME</div><div>VERDICT</div><div>SCORE</div><div>DATE</div>
        </div>
        {HISTORY.map((h, i) => {
          const vc = verdictColor(h.v);
          const score = RESULTS[h.v].score;
          return (
            <div key={i} onClick={() => onOpen(h.v)} style={{
              display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1.4fr",
              padding: "16px 20px", borderBottom: i < HISTORY.length - 1 ? `1px solid ${C.border}` : "none",
              cursor: "pointer", alignItems: "center", transition: "all 0.2s",
            }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#0c1838")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div>
                <div style={{ fontSize: 13, color: C.text, fontFamily: MONO, fontWeight: 500 }}>{h.filename}</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 4, letterSpacing: 0.3 }}>{h.institution}</div>
              </div>
              <div>
                <span style={{
                  fontSize: 10, padding: "5px 12px", borderRadius: 999,
                  background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg, letterSpacing: 1.5, fontWeight: 700, fontFamily: MONO,
                }}>{h.v}</span>
              </div>
              <div style={{ fontSize: 16, color: vc.fg, fontWeight: 700, fontFamily: MONO }}>{score}</div>
              <div style={{ fontSize: 11, color: C.muted, letterSpacing: 0.5, fontFamily: MONO }}>{h.date}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
