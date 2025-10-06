# app.py
# Gradio + OpenAI chatbot with optional PDF RAG (Colab-friendly)

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gradio as gr
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------
# Helpers: API key + OpenAI
# ---------------------------

def set_api_key_from_textbox(k: str) -> str:
    """Called when the user types a key in the UI textbox."""
    k = (k or "").strip()
    if k:
        os.environ["OPENAI_API_KEY"] = k
        return "API key set for this session."
    return "No API key provided. Set one in the box above or via environment."

def get_client():
    """Create the OpenAI client at call-time so Colab env vars are picked up."""
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "In Colab, run:\n"
            "  from google.colab import userdata\n"
            "  import os; os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API')\n"
            "Or paste the key in the UI field at the top."
        )
    return OpenAI(api_key=key)

# ---------------------------
# RAG store (FAISS + MiniLM)
# ---------------------------

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: Optional[SentenceTransformer] = None

@dataclass
class VectorStore:
    index: Optional[faiss.IndexFlatIP] = None
    texts: List[str] = None
    embeddings: Optional[np.ndarray] = None

VS = VectorStore(index=None, texts=[], embeddings=None)

def _ensure_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL_NAME)

def _chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    out, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i + chunk_size]))
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
        VS.index, VS.texts, VS.embeddings = None, [], None
        return "No text found in PDFs."

    _ensure_embedder()
    chunks = _chunk("\n".join(all_text))
    embs = _embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype("float32"))

    VS.index, VS.texts, VS.embeddings = index, chunks, embs
    return f"Indexed {len(chunks)} chunks."

def search_ctx(q: str, k: int = 4) -> List[str]:
    if VS.index is None or not VS.texts:
        return []
    _ensure_embedder()
    qemb = _embedder.encode([q], normalize_embeddings=True).astype("float32")
    D, I = VS.index.search(qemb, k)
    return [VS.texts[i] for i in I[0] if i != -1]

# ---------------------------
# Chat logic
# ---------------------------

def system_prompt(domain: str) -> str:
    s = "You are a helpful teaching assistant. Be concise, show steps when asked."
    if domain.strip():
        s += f" Focus on: {domain.strip()}."
    return s

def call_llm(messages, model="gpt-4o-mini", temperature=0.2) -> str:
    client = get_client()  # <- fetches key at call-time
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def chat_response(
    user_msg: str,
    history: List[Tuple[str, str]],
    use_rag: bool,
    domain: str,
    model: str,
    temperature: float,
) -> Tuple[List[Tuple[str, str]], str]:
    context = ""
    if use_rag:
        passages = search_ctx(user_msg, k=4)
        if passages:
            context = "Use this context if helpful:\n" + "\n\n".join(passages)

    msgs = [{"role": "system", "content": system_prompt(domain)}]
    if context:
        msgs.append({"role": "system", "content": context})

    # short history for continuity
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
    VS.index, VS.texts, VS.embeddings = None, [], None
    return "Index cleared."

# ---------------------------
# UI (Gradio)
# ---------------------------

with gr.Blocks(title="Course Chatbot (RAG + OpenAI)") as demo:
    gr.Markdown("# 🧠 Course Chatbot\nUpload PDFs (slides/notes) and ask questions.")

    with gr.Row():
        api_box = gr.Textbox(
            label="OPENAI_API_KEY (paste here if not set via environment)",
            placeholder="sk-****************",
            type="password",
        )
        model = gr.Dropdown(
            label="Model",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            value="gpt-4o-mini"
        )
        temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

    domain = gr.Textbox(label="Domain focus (e.g., 'Logistic Regression, LDA, Metrics')", value="")
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

    # when user types a key in the box, save to env
    api_box.change(fn=set_api_key_from_textbox, inputs=[api_box], outputs=[])

    send.click(
        fn=chat_response,
        inputs=[msg, chat, use_rag, domain, model, temp],
        outputs=[chat, msg],
    )

if __name__ == "__main__":
    # share=True prints a public URL in Colab output
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
