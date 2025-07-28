import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from typing import Tuple, Optional
from db import get_connection

DB_PATH = "girlfriend.db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = "models"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
EMBED_DIM = 384

# Initialize models only once (safe and fast)
EMBEDDER = SentenceTransformer(EMBED_MODEL)
LLM = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    model_file=MODEL_FILE,
    model_type="mistral",
    max_new_tokens=512,
    temperature=0.7,
    stream=False
)


def load_embeddings() -> Tuple[list[int], np.ndarray]:
    con = get_connection()
    rows = con.execute("SELECT id, embedding FROM msgs").fetchall()
    con.close()
    if not rows:
        return [], np.zeros((0, EMBED_DIM), dtype="float32")
    ids = [r[0] for r in rows]
    mats = [np.frombuffer(r[1], dtype="float32") for r in rows]
    return ids, np.vstack(mats)


def explain(emotion: str, top_k: int = 8) -> Tuple[str, Optional[int]]:
    """
    Returns (answer_text, top_msg_id).
    top_msg_id is None if there was no data.
    """
    IDS, EMBEDS = load_embeddings()  # always fresh

    if EMBEDS.shape[0] == 0:
        return "No messages ingested yet — please upload text or PDF first.", None

    # Embed the question
    question = f"Why is my girlfriend {emotion}? Cite evidence."
    qvec = EMBEDDER.encode(question).astype("float32")

    # Similarity search
    sims = EMBEDS.dot(qvec)
    top_idxs = np.argsort(-sims)[:top_k]

    if len(top_idxs) == 0:
        return "No relevant snippets found.", None

    # Filter out blocked examples
    con = sqlite3.connect(DB_PATH)
    blocked_ids = set(
        row[0] for row in con.execute(
            "SELECT text_id FROM feedback WHERE res IN ('never', 'it_is')"
        )
    )

    con.close()

    valid_idxs = [i for i in top_idxs if IDS[i] not in blocked_ids]
    if not valid_idxs:
        return "No relevant snippets found (or all were rejected before).", None

    first_id = IDS[valid_idxs[0]]

    # Retrieve context
    conn, snippets = sqlite3.connect(DB_PATH), []
    for idx in valid_idxs:
        row = conn.execute("SELECT text FROM msgs WHERE id=?", (IDS[idx],)).fetchone()
        if row:
            snippets.append(f"- {row[0]}")
    conn.close()

    context = "\n".join(snippets)
    prompt = f"Context:\n{context}\n\nQ:{question}\nA:"

    token_ids = LLM.tokenize(prompt)
    out_ids = list(LLM.generate(token_ids))
    response = LLM.detokenize(out_ids)

    return response.strip(), first_id
