# app.py
# Gradio + OpenAI chatbot with optional PDF RAG
# Works in Colab and locally. Launch: python app.py  (or) in Colab: !python app.py

import os
import gradio as gr
from typing import List, Tuple, Optional
from dataclasses import dataclass

# --- OpenAI client (new SDK style) ---
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Install dependencies first: pip install -r requirements.txt")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- RAG bits ---
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

EMB_MODEL_NAME = "all-MiniLM-L6-v2"   # small & fast
embedder: Optional[SentenceTransformer] = None

@dataclass
class VectorStore:
    index: Optional[faiss.IndexFlatIP] = None
    texts: List[str] = None
    embeddings: Optional[np.ndarray] = None

vs = VectorStore(index=None, texts=[], embeddings=None)


def ensure_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMB_MODEL_NAME)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks


def build_vectorstore_from_pdfs(pdf_files: List[gr.File]) -> Tuple[int, str]:
    """Return (num_chunks, status_msg)."""
    vs.texts = []
    all_text = []

    for f in pdf_files:
        reader = PdfReader(f.name)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                all_text.append(t)

    big = "\n".join(all_text)
    if not big.strip():
        vs.index = None
        vs.texts = []
        vs.embeddings = None
        return 0, "No text extracted. Check PDFs."

    chunks = chunk_text(big)
    ensure_embedder()
    embs = embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype("float32"))

    vs.index = index
    vs.texts = chunks
    vs.embeddings = embs
    return len(chunks), f"Indexed {len(chunks)} chunks."


def search(query: str, k: int = 4) -> List[str]:
    """Return top-k chunk strings."""
    if vs.index is None or not vs.texts:
        return []
    ensure_embedder()
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = vs.index.search(q, k)
    hits = [vs.texts[int(i)] for i in I[0] if i != -1]
    return hits


def format_system_prompt(domain: str) -> str:
    base = (
        "You are a helpful teaching assistant. "
        "Be concise, cite key concepts when relevant, and show steps when asked."
    )
    if domain:
        base += f" Focus on the topic: {domain}."
    return base


def call_llm(messages, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def respond(
    user_msg: str,
    chat_history: List[Tuple[str, str]],
    use_rag: bool,
    domain_focus: str,
    model_name: str,
    temperature: float,
) -> Tuple[List[Tuple[str, str]], str]:
    # Gather RAG context if enabled
    context = ""
    if use_rag:
        passages = search(user_msg, k=4)
        if passages:
            joined = "\n\n".join(passages)
            context = f"Use the following context if helpful:\n{joined}\n\n"

    system_prompt = format_system_prompt(domain_focus)
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})

    # Include short chat history for continuity
    for u, a in chat_history[-6:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_msg})

    try:
        answer = call_llm(messages, model=model_name, temperature=temperature)
    except Exception as e:
        answer = f"LLM error: {e}"

    chat_history = chat_history + [(user_msg, answer)]
    return chat_history, ""


def clear_index():
    vs.index = None
    vs.texts = []
    vs.embeddings = None
    return "Cleared index."


# --------- UI ---------
with gr.Blocks(title="Course Chatbot (RAG + OpenAI)") as demo:
    gr.Markdown("# 🧠 Course Chatbot\nAsk questions, upload PDFs for context, and chat.\n")
    with gr.Row():
        api = gr.Textbox(
            label="OPENAI_API_KEY (optional here if set in environment)",
            type="password",
            placeholder="sk-...",
        )
        model = gr.Dropdown(
            label="Model",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            value="gpt-4o-mini",
        )
        temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

    domain = gr.Textbox(label="Domain focus (e.g., 'Logistic Regression, LDA, Metrics')", value="")
    use_rag = gr.Checkbox(label="Use PDF knowledge (RAG)", value=True)

    with gr.Accordion("📄 Upload PDFs for RAG", open=False):
        pdfs = gr.File(file_count="multiple", file_types=[".pdf"], label="PDFs")
        build_btn = gr.Button("Build Index")
        status = gr.Markdown("")

        def _on_build(files):
            if not files:
                return "Upload one or more PDFs first."
            n, msg = build_vectorstore_from_pdfs(files)
            return f"{msg}"

        build_btn.click(_on_build, inputs=[pdfs], outputs=[status])
        gr.Button("Clear Index").click(lambda: clear_index(), outputs=[status])

    chat = gr.Chatbot(type="messages", height=400)
    msg = gr.Textbox(label="Your message", placeholder="Ask me anything…")
    send = gr.Button("Send")

    def _set_key(k):
        if k and k.strip():
            os.environ["OPENAI_API_KEY"] = k.strip()
        return "API key set in this session." if k else "Using environment variable if available."

    api.change(_set_key, inputs=[api], outputs=[])

    send.click(
        respond,
        inputs=[msg, chat, use_rag, domain, model, temp],
        outputs=[chat, msg],
    )

if __name__ == "__main__":
    # If running in Colab, this prints a share URL in the cell output.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
