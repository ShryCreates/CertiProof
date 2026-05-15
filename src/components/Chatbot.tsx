import { useState, useRef, useEffect, CSSProperties } from "react";
import { streamChat } from "@/lib/api";

// ── Design tokens (match Index.tsx) ──────────────────────────────────────────
const C = {
  bg:       "#05080f",
  card:     "rgba(14,20,40,0.95)",
  border:   "rgba(59,123,248,0.15)",
  borderHi: "rgba(59,123,248,0.35)",
  text:     "#dce8ff",
  muted:    "#4a5a8a",
  blue:     "#3b7bf8",
  blueGlow: "rgba(59,123,248,0.25)",
  green:    "#10b981",
  purple:   "#a855f7",
  userBg:   "rgba(59,123,248,0.12)",
  botBg:    "rgba(255,255,255,0.04)",
};
const FONT = "'Space Grotesk', 'Inter', system-ui, sans-serif";
const MONO = "'JetBrains Mono', 'Courier New', monospace";

interface Message {
  role:    "user" | "bot";
  content: string;
  loading?: boolean;
}

interface ChatbotProps {
  verificationId?: string | null;
}

const SUGGESTIONS = [
  "What does the forgery score mean?",
  "Why was this certificate flagged?",
  "What is ELA analysis?",
  "How is the trust score calculated?",
  "Is IIT Bombay in your database?",
  "What does NAAC A++ mean?",
];

export default function Chatbot({ verificationId }: ChatbotProps) {
  const [open, setOpen]         = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: "bot", content: "Hi! I'm the CertiProof AI Assistant. Ask me anything about certificate verification, forensic analysis, or the result you're viewing." },
  ]);
  const [input, setInput]       = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef               = useRef<HTMLDivElement>(null);
  const inputRef                = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 100);
  }, [open]);

  const send = async (text: string) => {
    const msg = text.trim();
    if (!msg || streaming) return;
    setInput("");

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: msg }]);

    // Add empty bot message that will be streamed into
    setMessages(prev => [...prev, { role: "bot", content: "", loading: true }]);
    setStreaming(true);

    await streamChat(
      msg,
      verificationId ?? null,
      // onToken — append each word
      (token) => {
        setMessages(prev => {
          const updated = [...prev];
          const last    = updated[updated.length - 1];
          if (last.role === "bot") {
            updated[updated.length - 1] = { ...last, content: last.content + token, loading: false };
          }
          return updated;
        });
      },
      // onDone
      () => {
        setStreaming(false);
        setMessages(prev => {
          const updated = [...prev];
          const last    = updated[updated.length - 1];
          if (last.role === "bot") updated[updated.length - 1] = { ...last, loading: false };
          return updated;
        });
      },
      // onError
      (err) => {
        setStreaming(false);
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "bot", content: `⚠ ${err}`, loading: false };
          return updated;
        });
      },
    );
  };

  return (
    <>
      <style>{`
        @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes blink  { 0%,100%{opacity:1} 50%{opacity:0} }
        .chat-send:hover  { background: #2563eb !important; }
        .chat-sugg:hover  { background: rgba(59,123,248,0.15) !important; border-color: ${C.blue} !important; }
        .chat-msg-user    { animation: fadeUp 0.2s ease; }
        .chat-msg-bot     { animation: fadeUp 0.2s ease; }
      `}</style>

      {/* Floating button */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          position: "fixed", bottom: 28, right: 28, zIndex: 1000,
          width: 56, height: 56, borderRadius: "50%",
          background: "linear-gradient(135deg, #3b7bf8, #6366f1)",
          border: "none", cursor: "pointer",
          boxShadow: `0 4px 24px ${C.blueGlow}, 0 0 0 ${open ? "3px" : "0px"} rgba(59,123,248,0.4)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 22, transition: "all 0.3s",
          animation: open ? "none" : "bounce 2s ease infinite",
        }}
        title="CertiProof AI Assistant"
      >
        {open ? "✕" : "🤖"}
      </button>

      {/* Chat panel */}
      {open && (
        <div style={{
          position: "fixed", bottom: 96, right: 28, zIndex: 999,
          width: 380, height: 560,
          background: C.card,
          border: `1px solid ${C.borderHi}`,
          borderRadius: 16,
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          display: "flex", flexDirection: "column",
          boxShadow: `0 24px 64px rgba(0,0,0,0.6), 0 0 0 1px rgba(59,123,248,0.1)`,
          animation: "fadeUp 0.25s ease",
          overflow: "hidden",
        }}>

          {/* Header */}
          <div style={{
            padding: "16px 18px", borderBottom: `1px solid ${C.border}`,
            background: "rgba(59,123,248,0.06)",
            display: "flex", alignItems: "center", gap: 12, flexShrink: 0,
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10, flexShrink: 0,
              background: "linear-gradient(135deg, rgba(59,123,248,0.3), rgba(168,85,247,0.2))",
              border: `1px solid ${C.borderHi}`,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
            }}>🤖</div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: C.text, fontFamily: FONT }}>CertiProof Assistant</div>
              <div style={{ fontSize: 10, color: C.green, letterSpacing: 1, fontFamily: MONO, display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, boxShadow: `0 0 6px ${C.green}` }} />
                {streaming ? "TYPING..." : "ONLINE · phi3"}
              </div>
            </div>
            {verificationId && (
              <div style={{ marginLeft: "auto", fontSize: 9, color: C.blue, letterSpacing: 1, fontFamily: MONO, padding: "3px 8px", background: "rgba(59,123,248,0.1)", borderRadius: 4, border: `1px solid ${C.border}` }}>
                RESULT CONTEXT
              </div>
            )}
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "16px 14px", display: "flex", flexDirection: "column", gap: 12 }}>
            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "chat-msg-user" : "chat-msg-bot"}
                style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
                {m.role === "bot" && (
                  <div style={{ width: 26, height: 26, borderRadius: 8, flexShrink: 0, marginRight: 8, marginTop: 2, background: "linear-gradient(135deg, rgba(59,123,248,0.2), rgba(168,85,247,0.15))", border: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13 }}>🤖</div>
                )}
                <div style={{
                  maxWidth: "78%", padding: "10px 14px", borderRadius: m.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
                  background: m.role === "user" ? C.userBg : C.botBg,
                  border: `1px solid ${m.role === "user" ? "rgba(59,123,248,0.25)" : C.border}`,
                  fontSize: 13, color: C.text, lineHeight: 1.65, fontFamily: FONT,
                }}>
                  {m.loading ? (
                    <div style={{ display: "flex", gap: 4, alignItems: "center", padding: "2px 0" }}>
                      {[0, 1, 2].map(j => (
                        <div key={j} style={{ width: 6, height: 6, borderRadius: "50%", background: C.blue, animation: `blink 1.2s ease ${j * 0.2}s infinite` }} />
                      ))}
                    </div>
                  ) : m.content}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Suggestions (only when no conversation yet) */}
          {messages.length <= 1 && (
            <div style={{ padding: "0 14px 10px", display: "flex", flexWrap: "wrap", gap: 6 }}>
              {SUGGESTIONS.slice(0, 4).map((s, i) => (
                <button key={i} className="chat-sugg" onClick={() => send(s)} style={{
                  fontSize: 10, padding: "5px 10px", borderRadius: 6, cursor: "pointer",
                  background: "rgba(255,255,255,0.03)", border: `1px solid ${C.border}`,
                  color: C.muted, fontFamily: FONT, transition: "all 0.2s", textAlign: "left",
                }}>{s}</button>
              ))}
            </div>
          )}

          {/* Input */}
          <div style={{ padding: "12px 14px", borderTop: `1px solid ${C.border}`, display: "flex", gap: 8, flexShrink: 0 }}>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && send(input)}
              placeholder="Ask anything about certificates..."
              disabled={streaming}
              style={{
                flex: 1, padding: "10px 14px",
                background: "rgba(255,255,255,0.04)",
                border: `1px solid ${input ? C.borderHi : C.border}`,
                borderRadius: 8, color: C.text, fontFamily: FONT, fontSize: 12,
                outline: "none", transition: "all 0.2s",
                opacity: streaming ? 0.6 : 1,
              }}
            />
            <button
              className="chat-send"
              onClick={() => send(input)}
              disabled={!input.trim() || streaming}
              style={{
                width: 40, height: 40, borderRadius: 8, flexShrink: 0,
                background: input.trim() && !streaming ? C.blue : "rgba(255,255,255,0.05)",
                border: "none", cursor: input.trim() && !streaming ? "pointer" : "not-allowed",
                color: "#fff", fontSize: 16, transition: "all 0.2s",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              {streaming ? (
                <svg width={14} height={14} viewBox="0 0 14 14" style={{ animation: "spin 0.8s linear infinite" }}>
                  <circle cx={7} cy={7} r={5} fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={2}/>
                  <path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round"/>
                </svg>
              ) : "↑"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
