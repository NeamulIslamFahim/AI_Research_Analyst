const API_BASE = process.env.REACT_APP_API_BASE || "";

async function handleResponse(res) {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Request failed");
  }
  return res.json();
}

export async function uploadReviewPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/review/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function reviewQA(question, paperText) {
  const res = await fetch(`${API_BASE}/api/review/qa`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, paper_text: paperText }),
  });
  return handleResponse(res);
}

export async function researchExplore(topic, chatHistory, focusTopic, useLive, forceRefresh) {
  const res = await fetch(`${API_BASE}/api/research/explore`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      chat_history: chatHistory || null,
      focus_topic: focusTopic || null,
      use_live: useLive ?? null,
      force_refresh: forceRefresh ?? null,
    }),
  });
  return handleResponse(res);
}


export async function assistantChat(prompt, chatHistory) {
  const res = await fetch(`${API_BASE}/api/assistant/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      chat_history: chatHistory || null,
    }),
  });
  return handleResponse(res);
}


export async function trainAssistant(force = false) {
  const res = await fetch(`${API_BASE}/api/assistant/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
  return handleResponse(res);
}


export async function generateReferences(topic) {
  const res = await fetch(`${API_BASE}/api/reference`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });
  return handleResponse(res);
}

export async function writerStep(userText, state) {
  const res = await fetch(`${API_BASE}/api/writer/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_text: userText, state: state || {} }),
  });
  return handleResponse(res);
}

export async function downloadPapers(topic) {
  const res = await fetch(`${API_BASE}/api/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });
  return handleResponse(res);
}
