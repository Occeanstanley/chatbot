# app.py — Gradio + OpenAI chatbot with PDF RAG (Windows-friendly: sklearn instead of FAISS)

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gradio as gr
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ----- API key helpers -----
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
        raise ValueError("OPENAI_API_KEY not set. Paste it in the UI or set env var.")
    return OpenAI(api_key=key)

# ----- RAG store (SentenceTransformer + sklearn KNN) -----
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: Optional[SentenceTransformer] = None

@dataclass
class VectorStore:
    nn: Optional[NearestNeighbors] = None
    texts: List[str] = None
    embeddings: Optional[np.ndarray] = None

VS = VectorStore(nn=None, texts=[], embeddings=None)

def _ensure_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL_NAME)

def _chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    out, i, step = [], 0, max(1, chunk_size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += step
    return out

def build_index_from_pdfs(files: List[gr.File]) -> str:
    all_text = []
    for f in files or []:
        reader = PdfReader(f.name)
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                all_text.append(t)
    if not all_text:
        VS.nn, VS.texts, VS.embeddings = None, [], None
        return "No text found."

    _ensure_embedder()
    chunks = _chunk("\n".join(all_text))
    embs = _embedder.encode(chunks, normalize_embeddings=True)
    # cosine sim via brute-force (metric='cosine' with 1 - cosine distance)
    nn = NearestNeighbors(n_neighbors=4, metric="cosine")
    nn.fit(embs)

    VS.nn, VS.texts, VS.embeddings = nn, chunks, embs
    return f"Indexed {len(chunks)} chunks."

def search_ctx(q: str, k: int = 4) -> List[str]:
    if VS.nn is None or not VS.texts:
        return []
    _ensure_embedder()
    qemb = _embedder.encode([q], normalize_embeddings=True)
    distances, indices = VS.nn.kneighbors(qemb, n_neighbors=min(k, len(VS.texts)))
    return [VS.texts[i] for i in indices[0]]

# ----- Chat logic -----
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
    VS.nn, VS.texts, VS.embeddings = None, [], None
    return "Index cleared."

# ----- UI -----
with gr.Blocks(title="Course Chatbot (RAG + OpenAI)") as demo:
    gr.Markdown("# 🧠 Course Chatbot\nUpload PDFs and ask questions.")

    with gr.Row():
        api_box = gr.Textbox(label="OPENAI_API_KEY", type="password", placeholder="sk-...")
        model = gr.Dropdown(choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], value="gpt-4o-mini", label="Model")
        temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

    domain = gr.Textbox(label="Domain focus (e.g., Logistic Regression, LDA, Metrics)", value="")
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
