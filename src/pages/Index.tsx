import { useEffect, useRef, useState, CSSProperties, DragEvent } from "react";

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
  <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1, marginBottom: 6 }}>{children}</div>;

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
        <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1, marginBottom: 4 }}>SIGNED IN</div>
        <div style={{ fontSize: 11, color: C.text, marginBottom: 12, wordBreak: "break-all" }}>{email}</div>
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
        <div style={{ fontSize: 11, color: C.muted, letterSpacing: 1, marginBottom: 14 }}>
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
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 1, textAlign: "center", marginTop: 8 }}>
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
          <div style={{ fontSize: 11, color: C.red, letterSpacing: 1, marginBottom: 10, fontWeight: "bold" }}>
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

      <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
        <button style={btnPrimary()}>📥 DOWNLOAD PDF REPORT</button>
        <button style={btnGhost()}>🔗 SHARE RESULT</button>
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

// ============ SCORE RING ============
function ScoreCard({ result, vc }: { result: Result; vc: { fg: string; bg: string } }) {
  const r = 46, c = 2 * Math.PI * r;
  const offset = c - (result.score / 100) * c;
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 24, textAlign: "center" }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1, marginBottom: 18 }}>VERDICT</div>
      <svg width={140} height={140} viewBox="0 0 120 120" style={{ transform: "rotate(-90deg)" }}>
        <circle cx={60} cy={60} r={r} fill="none" stroke={C.border} strokeWidth={8} />
        <circle cx={60} cy={60} r={r} fill="none" stroke={vc.fg} strokeWidth={8}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={offset}
          style={{ transition: "stroke-dasharray 1s ease, stroke-dashoffset 1s ease" }} />
      </svg>
      <div style={{ marginTop: -98, height: 98, display: "flex", flexDirection: "column", justifyContent: "center", pointerEvents: "none" }}>
        <div style={{ fontSize: 38, color: vc.fg, fontWeight: 700, letterSpacing: 1, fontFamily: MONO }}>{result.score}</div>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 3, marginTop: 4, fontFamily: MONO }}>TRUST SCORE</div>
      </div>
      <div style={{ marginTop: 18 }}>
        <span style={{
          display: "inline-block", padding: "8px 18px", borderRadius: 999,
          background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg,
          fontSize: 12, letterSpacing: 2, fontWeight: "bold",
        }}>{result.verdict}</span>
      </div>
      <div style={{ marginTop: 22, padding: 12, background: C.bg, borderRadius: 6, border: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1, marginBottom: 6 }}>INSTITUTION MATCH</div>
        <div style={{ fontSize: 13, color: result.institution_match ? C.green : C.red, letterSpacing: 1 }}>
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
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 1, marginBottom: 16 }}>SCORE CONTRIBUTION (WEIGHTED)</div>
      {bars.map((b) => (
        <div key={b.label} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: C.text, letterSpacing: 1, width: 80 }}>{b.label}</div>
          <div style={{ flex: 1, height: 18, background: C.bg, borderRadius: 4, overflow: "hidden", border: `1px solid ${C.border}` }}>
            <div style={{
              width: `${b.weight}%`, height: "100%",
              background: b.color, transition: "width 0.8s ease",
            }} />
          </div>
          <div style={{ fontSize: 11, color: b.color, width: 44, textAlign: "right", letterSpacing: 1 }}>{b.weight}%</div>
        </div>
      ))}
    </div>
  );
}

function FieldsCard({ result }: { result: Result }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 1, marginBottom: 14 }}>EXTRACTED FIELDS</div>
      {Object.entries(result.fields).map(([k, v], i, arr) => (
        <div key={k} style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "10px 0", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none",
        }}>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 1 }}>{k}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ fontSize: 12, color: C.text }}>{v.value}</div>
            <div style={{ fontSize: 10, color: confColor(v.confidence), letterSpacing: 1, minWidth: 36, textAlign: "right" }}>
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
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 1, marginBottom: 12 }}>🧠 LLM REASONING</div>
      <div style={{ fontSize: 12, color: C.text, lineHeight: 1.7 }}>{result.reasoning}</div>
    </div>
  );
}

function HeatmapCard() {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 22 }}>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 1, marginBottom: 12 }}>🗺 GRADCAM HEATMAP</div>
      <div style={{
        height: 120, borderRadius: 6, border: `1px dashed ${C.border}`,
        background: "linear-gradient(135deg, #0a1530 0%, #1a0a30 50%, #2a0a0a 100%)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, color: C.muted, letterSpacing: 1, marginBottom: 12,
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
          fontSize: 10, color: C.muted, letterSpacing: 2,
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
                <div style={{ fontSize: 12, color: C.text }}>{h.filename}</div>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 3, letterSpacing: 0.5 }}>{h.institution}</div>
              </div>
              <div>
                <span style={{
                  fontSize: 10, padding: "4px 10px", borderRadius: 999,
                  background: vc.bg, border: `1px solid ${vc.fg}`, color: vc.fg, letterSpacing: 1, fontWeight: "bold",
                }}>{h.v}</span>
              </div>
              <div style={{ fontSize: 14, color: vc.fg, fontWeight: "bold" }}>{score}</div>
              <div style={{ fontSize: 11, color: C.muted, letterSpacing: 0.5 }}>{h.date}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
