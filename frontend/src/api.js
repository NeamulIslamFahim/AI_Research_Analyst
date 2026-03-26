const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

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

export async function researchExplore(topic, chatHistory, focusTopic) {
  const res = await fetch(`${API_BASE}/api/research/explore`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      chat_history: chatHistory || null,
      focus_topic: focusTopic || null,
    }),
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

export async function generateReferences(topic) {
  const res = await fetch(`${API_BASE}/api/reference`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });
  return handleResponse(res);
}

export async function writerStep(step, userText) {
  const res = await fetch(`${API_BASE}/api/writer/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ step, user_text: userText }),
  });
  return handleResponse(res);
}
