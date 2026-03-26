import React, { useMemo, useState } from "react";
import "./App.css";
import { uploadReviewPdf, reviewQA, researchExplore, writerStep, downloadPapers } from "./api";

const MODES = ["Research Explorer", "Research Paper Reviewer", "Research Paper Writer"];

function isFollowupPrompt(text) {
  const t = (text || "").trim().toLowerCase();
  if (!t) return false;
  const phrases = [
    "more",
    "continue",
    "tell me more",
    "more on it",
    "more on this",
    "more on that",
    "more research",
    "more research on it",
    "expand",
    "elaborate",
    "same topic",
    "same one",
    "go deeper",
    "go deeper on it",
    "add more",
  ];
  if (phrases.includes(t)) return true;
  if (t.split(" ").length <= 4 && ["more", "continue", "expand", "elaborate", "same"].some((k) => t.includes(k))) {
    return true;
  }
  return false;
}

function formatChatHistory(messages) {
  return messages
    .map((m) => {
      const label = m.role === "user" ? "User" : "Assistant";
      const content =
        m.role === "user"
          ? typeof m.effectiveQuery === "string"
            ? m.effectiveQuery
            : typeof m.content === "string"
            ? m.content
            : ""
          : typeof m.content === "string"
          ? m.content
          : "";
      return `${label}: ${content}`;
    })
    .join("\n");
}

function formatReviewText(review) {
  if (!review || typeof review !== "object") return String(review || "");
  const parts = [
    "Here is a structured peer review of the paper:",
    `Strengths: ${review.strengths || ""}`,
    `Weaknesses: ${review.weaknesses || ""}`,
    `Novelty: ${review.novelty || ""}`,
    `Technical Correctness: ${review.technical_correctness || ""}`,
    `Reproducibility: ${review.reproducibility || ""}`,
    `Recommendation: ${review.recommendation || ""}`,
    `Suggested Venue: ${review.suggested_venue || ""}`,
  ];
  return parts.filter((p) => p && !p.endsWith(": ")).join("\n\n");
}

function ResearchResult({ result }) {
  if (!result || typeof result !== "object") return null;
  const table = Array.isArray(result.table) ? result.table : [];
  const gaps = Array.isArray(result.research_gaps) ? result.research_gaps : [];
  const steps = Array.isArray(result.generated_idea_steps) ? result.generated_idea_steps : [];
  return (
    <div className="research-result">
      {result.assistant_reply && <p>{result.assistant_reply}</p>}
      <h4>Results Table</h4>
      {table.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>paper_name</th>
                <th>paper_url</th>
                <th>authors_name</th>
                <th>summary_full_paper</th>
                <th>proposed_model_or_approach</th>
                <th>source</th>
              </tr>
            </thead>
            <tbody>
              {table.map((row, i) => (
                <tr key={i}>
                  <td>{row.paper_name}</td>
                  <td>
                    {row.paper_url ? (
                      <a href={row.paper_url} target="_blank" rel="noreferrer">
                        {row.paper_url}
                      </a>
                    ) : (
                      ""
                    )}
                  </td>
                  <td>{row.authors_name}</td>
                  <td>{row.summary_full_paper}</td>
                  <td>{row.proposed_model_or_approach}</td>
                  <td>{row.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No table rows returned.</p>
      )}
      <h4>Research Gaps</h4>
      {gaps.length ? (
        <ul>
          {gaps.map((g, i) => (
            <li key={i}>{g}</li>
          ))}
        </ul>
      ) : (
        <p>Not provided.</p>
      )}
      <h4>Generated Idea</h4>
      <p>{result.generated_idea || "Not provided."}</p>
      {steps.length ? (
        <>
          <h4>Implementation Steps</h4>
          <ul>
            {steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

export default function App() {
  const [sessions, setSessions] = useState([
    {
      id: "chat-1",
      title: "New Chat",
      mode: MODES[0],
      messages: [],
      createdAt: Date.now(),
    },
  ]);
  const [currentSessionId, setCurrentSessionId] = useState("chat-1");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [paperTextBySession, setPaperTextBySession] = useState({});
  const [writerStepBySession, setWriterStepBySession] = useState({});
  const [editIndex, setEditIndex] = useState(null);
  const [editText, setEditText] = useState("");

  const currentSession = sessions.find((s) => s.id === currentSessionId) || sessions[0];
  const mode = currentSession.mode;
  const messages = currentSession.messages;
  const paperText = paperTextBySession[currentSessionId] || "";
  const writerStep = writerStepBySession[currentSessionId] || 0;

  const lastTopic = useMemo(() => {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    return lastUser ? lastUser.effectiveQuery || lastUser.content : "";
  }, [messages]);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    updateMessages((m) => [...m, { role: "assistant", content: "Loading…", type: "loading" }]);
    try {
      const res = await uploadReviewPdf(file);
      setPaperTextBySession((m) => ({ ...m, [currentSessionId]: res.paper_text || "" }));
      const reviewText = res.review_text || formatReviewText(res.review);
      updateMessages((m) =>
        replaceOrAppendAssistant(m, m.length - 1, {
          role: "assistant",
          content: reviewText,
          type: "review",
        })
      );
    } catch (err) {
      updateMessages((m) =>
        replaceOrAppendAssistant(m, m.length - 1, {
          role: "assistant",
          content: String(err),
          type: "text",
        })
      );
    } finally {
      setLoading(false);
    }
  }

  function updateMessages(updater) {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === currentSessionId ? { ...s, messages: updater(s.messages) } : s
      )
    );
  }

  function updateSession(patch) {
    setSessions((prev) =>
      prev.map((s) => (s.id === currentSessionId ? { ...s, ...patch } : s))
    );
  }

  async function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    const followup = isFollowupPrompt(trimmed);
    const effectiveQuery = followup && lastTopic ? `${lastTopic} (continue with more detail)` : trimmed;
    const userMessage = { role: "user", content: trimmed, effectiveQuery };
    setInput("");
    if (messages.length === 0) {
      const title = trimmed.length > 40 ? `${trimmed.slice(0, 40)}...` : trimmed;
      updateSession({ title: title || "New Chat" });
    }

    if (mode === "Research Paper Writer") {
      setLoading(true);
      try {
        const res = await writerStep(writerStep, trimmed);
        const replies = Array.isArray(res.messages) ? res.messages : [];
        updateMessages((m) => [
          ...m,
          userMessage,
          ...replies.map((r) => ({ role: "assistant", content: r, type: "text" })),
        ]);
        setWriterStepBySession((m) => ({
          ...m,
          [currentSessionId]: typeof res.next_step === "number" ? res.next_step : 0,
        }));
      } catch (err) {
        updateMessages((m) => [...m, { role: "assistant", content: String(err), type: "text" }]);
      } finally {
        setLoading(false);
      }
      return;
    }

    updateMessages((m) => [...m, userMessage, { role: "assistant", content: "Loading…", type: "loading" }]);
    setLoading(true);
    try {
      if (mode === "Research Paper Reviewer") {
        if (!paperText) {
          updateMessages((m) => replaceOrAppendAssistant(m, m.length - 1, { role: "assistant", content: "Please upload a PDF first.", type: "text" }));
        } else {
          const res = await reviewQA(trimmed, paperText);
          updateMessages((m) => replaceOrAppendAssistant(m, m.length - 1, { role: "assistant", content: res.answer || "No answer found.", type: "text" }));
        }
      } else {
        const history = formatChatHistory(messages);
        const res = await researchExplore(effectiveQuery, history, followup ? lastTopic || null : null);
        updateMessages((m) => replaceOrAppendAssistant(m, m.length - 1, { role: "assistant", content: res, type: "research" }));
        // Fire-and-forget: download PDFs to update vector DB for faster future responses.
        downloadPapers(effectiveQuery).catch(() => {});
      }
    } catch (err) {
      updateMessages((m) => replaceOrAppendAssistant(m, m.length - 1, { role: "assistant", content: String(err), type: "text" }));
    } finally {
      setLoading(false);
    }
  }

  async function handleRegen(idx) {
    if (loading) return;
    const msg = messages[idx];
    const userText = msg.effectiveQuery || msg.content;
    updateMessages((m) => replaceOrAppendAssistant(m, idx, { role: "assistant", content: "Loading…", type: "loading" }));
    setLoading(true);
    try {
      let newAssistant;
      if (mode === "Research Paper Reviewer") {
        if (!paperText) {
          newAssistant = { role: "assistant", content: "Please upload a PDF first.", type: "text" };
        } else {
          const res = await reviewQA(userText, paperText);
          newAssistant = { role: "assistant", content: res.answer || "No answer found.", type: "text" };
        }
      } else {
        const history = formatChatHistory(messages);
        const res = await researchExplore(userText, history, lastTopic || null);
        newAssistant = { role: "assistant", content: res, type: "research" };
        downloadPapers(userText).catch(() => {});
      }
      updateMessages((m) => replaceOrAppendAssistant(m, idx, newAssistant));
    } catch (err) {
      updateMessages((m) =>
        replaceOrAppendAssistant(m, idx, { role: "assistant", content: String(err), type: "text" })
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleEditSave(idx) {
    if (loading) return;
    const newText = editText.trim();
    setEditIndex(null);
    setEditText("");
    if (!newText) return;
    const updated = messages.map((m, i) =>
      i === idx ? { ...m, content: newText, effectiveQuery: newText } : m
    );
    updateSession({ messages: updated });
    await handleRegen(idx);
  }

  function replaceOrAppendAssistant(list, userIdx, assistantMsg) {
    // Prefer replacing a loading placeholder if present.
    const loadingIdx = list.findIndex((m) => m.role === "assistant" && m.type === "loading");
    if (loadingIdx !== -1) {
      const copy = [...list];
      copy[loadingIdx] = assistantMsg;
      return copy;
    }
    const nextAssistantIdx = list.findIndex((m, i) => i > userIdx && m.role === "assistant");
    if (nextAssistantIdx !== -1) {
      const copy = [...list];
      copy[nextAssistantIdx] = assistantMsg;
      return copy;
    }
    return [...list, assistantMsg];
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <button
            className="new-chat"
            onClick={() => {
              const id = `chat-${sessions.length + 1}`;
              const newSession = {
                id,
                title: "New Chat",
                mode,
                messages: [],
                createdAt: Date.now(),
              };
              setSessions((s) => [newSession, ...s]);
              setCurrentSessionId(id);
              setPaperTextBySession((m) => ({ ...m, [id]: "" }));
              setWriterStepBySession((m) => ({ ...m, [id]: 0 }));
            }}
          >
            New Chat
          </button>
          <select
            value={mode}
            onChange={(e) => {
              const nextMode = e.target.value;
              if (messages.length === 0) {
                updateSession({ mode: nextMode });
                setWriterStepBySession((m) => ({ ...m, [currentSessionId]: 0 }));
                setPaperTextBySession((m) => ({ ...m, [currentSessionId]: "" }));
              } else {
                const id = `chat-${sessions.length + 1}`;
                const newSession = {
                  id,
                  title: "New Chat",
                  mode: nextMode,
                  messages: [],
                  createdAt: Date.now(),
                };
                setSessions((s) => [newSession, ...s]);
                setCurrentSessionId(id);
                setWriterStepBySession((m) => ({ ...m, [id]: 0 }));
                setPaperTextBySession((m) => ({ ...m, [id]: "" }));
              }
            }}
          >
            {MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div className="session-list">
          {sessions.map((s) => (
            <button
              key={s.id}
              className={`session-item ${s.id === currentSessionId ? "active" : ""}`}
              onClick={() => {
                setCurrentSessionId(s.id);
              }}
            >
              <div className="session-title">{s.title}</div>
              <div className="session-meta">{s.mode}</div>
            </button>
          ))}
        </div>
      </aside>

      <div className="main">
        <header>
          <h1>AI Research Assistant</h1>
          <div className="mode-pill">{mode}</div>
        </header>

      {mode === "Research Paper Reviewer" && (
        <section className="uploader">
          <label className="upload-btn">
            Upload PDF for review
            <input type="file" accept="application/pdf" onChange={handleUpload} />
          </label>
          {paperText ? <span className="upload-ok">PDF loaded</span> : null}
        </section>
      )}

      <main className="chat">
        {messages.map((msg, idx) => (
          <div key={idx} className={`msg ${msg.role}`}>
            <div className="bubble">
              {msg.role === "assistant" && msg.type === "loading" ? (
                <div className="loading">Loading…</div>
              ) : msg.role === "assistant" && msg.type === "research" ? (
                <ResearchResult result={msg.content} />
              ) : (
                <pre>{String(msg.content)}</pre>
              )}
            </div>
            {msg.role === "user" && mode !== "Research Paper Writer" && (
              <div className="actions">
                {editIndex === idx ? null : (
                  <>
                    <button onClick={() => { setEditIndex(idx); setEditText(msg.content); }}>Edit</button>
                    <button onClick={() => handleRegen(idx)}>Regen</button>
                  </>
                )}
              </div>
            )}
            {editIndex === idx && (
              <div className="edit">
                <textarea value={editText} onChange={(e) => setEditText(e.target.value)} />
                <div className="edit-actions">
                  <button onClick={() => handleEditSave(idx)}>Save</button>
                  <button onClick={() => { setEditIndex(null); setEditText(""); }}>Cancel</button>
                </div>
              </div>
            )}
          </div>
        ))}
      </main>

      <footer>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your message"
          onKeyDown={(e) => (e.key === "Enter" ? handleSend() : null)}
        />
        <button onClick={handleSend} disabled={loading}>
          {loading ? "Working..." : "Send"}
        </button>
      </footer>
      </div>
    </div>
  );
}
