# app.py — Windows-friendly Gradio chatbot with optional PDF RAG
# Uses OpenAI embeddings + scikit-learn NearestNeighbors (no FAISS)

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gradio as gr
import numpy as np
from pypdf import PdfReader
from sklearn.neighbors import NearestNeighbors

# ---------- API key ----------
def set_api_key_from_textbox(k: str) -> str:
    k = (k or "").strip()
    if k:
        os.environ["OPENAI_API_KEY"] = k
        return "API key set for this session."
    return "No API key provided."

def get_client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Paste it in the UI or export the env var.")
    return OpenAI(api_key=key)

# ---------- Embeddings & store ----------
EMBED_MODEL = "text-embedding-3-small"  # fast + inexpensive

@dataclass
class VectorStore:
    nn: Optional[NearestNeighbors] = None
    texts: List[str] = None
    vectors: Optional[np.ndarray] = None

VS = VectorStore(nn=None, texts=[], vectors=None)

def _chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    out, i, step = [], 0, max(1, chunk_size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += step
    return out

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts with OpenAI and L2-normalize for cosine similarity."""
    if not texts:
        return np.zeros((0, 1536), dtype="float32")
    client = get_client()
    out = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
        out.extend(vecs)
    X = np.vstack(out)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def build_index_from_pdfs(files: List[gr.File]) -> str:
    all_text = []
    for f in files or []:
        reader = PdfReader(f.name)
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                all_text.append(t)
    if not all_text:
        VS.nn, VS.texts, VS.vectors = None, [], None
        return "No text found."

    chunks = _chunk("\n".join(all_text))
    vectors = _embed_texts(chunks)
    nn = NearestNeighbors(n_neighbors=4, metric="cosine")
    nn.fit(vectors)

    VS.nn, VS.texts, VS.vectors = nn, chunks, vectors
    return f"Indexed {len(chunks)} chunks."

def search_ctx(q: str, k: int = 4) -> List[str]:
    if VS.nn is None or not VS.texts:
        return []
    qvec = _embed_texts([q])
    dist, idx = VS.nn.kneighbors(qvec, n_neighbors=min(k, len(VS.texts)))
    return [VS.texts[i] for i in idx[0]]

# ---------- Chat logic ----------
def system_prompt(domain: str) -> str:
    s = "You are a helpful teaching assistant. Be concise; show steps when asked."
    if domain.strip():
        s += f" Focus on: {domain.strip()}."
    return s

def call_llm(messages, model="gpt-4o-mini", temperature=0.2) -> str:
    client = get_client()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content

def chat_response(user_msg, history, use_rag, domain, model, temperature):
    context = ""
    if use_rag:
        hits = search_ctx(user_msg, k=4)
        if hits:
            context = "Use this context if helpful:\n" + "\n\n".join(hits)

    msgs = [{"role": "system", "content": system_prompt(domain)}]
    if context:
        msgs.append({"role": "system", "content": context})
    for u, a in history[-6:]:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})

    try:
        answer = call_llm(msgs, model=model, temperature=temperature)
    except Exception as e:
        answer = f"⚠️ LLM error: {e}"
    history = history + [(user_msg, answer)]
    return history, ""

def clear_index():
    VS.nn, VS.texts, VS.vectors = None, [], None
    return "Index cleared."

# ---------- UI ----------
with gr.Blocks(title="Course Chatbot (RAG + OpenAI)") as demo:
    gr.Markdown("# 🧠 Course Chatbot\nUpload PDFs and ask questions.")

    with gr.Row():
        api_box = gr.Textbox(label="OPENAI_API_KEY", type="password", placeholder="sk-...")
        model = gr.Dropdown(choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], value="gpt-4o-mini", label="Model")
        temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

    domain = gr.Textbox(label="Domain focus (e.g., Logistic Regression, LDA, Metrics)")
    use_rag = gr.Checkbox(label="Use PDF knowledge (RAG)", value=True)

    with gr.Accordion("📄 Upload PDFs for RAG", open=False):
        pdfs = gr.File(file_count="multiple", file_types=[".pdf"], label="PDFs")
        build_btn = gr.Button("Build Index")
        status = gr.Markdown("")
        build_btn.click(fn=build_index_from_pdfs, inputs=[pdfs], outputs=[status])
        gr.Button("Clear Index").click(fn=clear_index, outputs=[status])

    chat = gr.Chatbot(type="messages", height=420)
    msg = gr.Textbox(label="Your message", placeholder="Ask me anything…")
    send = gr.Button("Send")

    api_box.change(fn=set_api_key_from_textbox, inputs=[api_box], outputs=[])

    send.click(fn=chat_response,
               inputs=[msg, chat, use_rag, domain, model, temp],
               outputs=[chat, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
