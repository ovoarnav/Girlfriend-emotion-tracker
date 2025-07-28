import fitz
import json
import sqlite3
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

DB_PATH = Path("girlfriend.db")
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMO_MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

embedder = SentenceTransformer(EMBEDDER_NAME)
tok = AutoTokenizer.from_pretrained(EMO_MODEL_NAME)
emo_model = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_NAME)
emo_pipe = pipeline("text-classification", model=emo_model,
                    tokenizer=tok, top_k=None,
                    function_to_apply="sigmoid")


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS msgs(
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL,
        emotion TEXT NOT NULL,
        score REAL NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback(
        text_id INTEGER,
        res TEXT CHECK(res IN ('never','maybe','it_is','more')),
        UNIQUE(text_id, res)
    )""")
    return con


def add_pdf(uploaded_file):
    con = _connect()
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            for line in page.get_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                _insert_text(con, line)
    con.commit()
    con.close()


def _insert_text(con, text):
    # Generate sentence embedding
    vec = embedder.encode(text).astype("float32")

    # Get emotion scores
    emos = emo_pipe(text)[0]  # list of {label, score}
    # Safety check in case emos is a string
    if isinstance(emos, str):
        emos = json.loads(emos)

    top = max(emos, key=lambda x: x["score"])
    # INSERT WITH EMBEDDING
    con.execute(
        "INSERT INTO msgs(text, embedding, emotion, score) VALUES (?, ?, ?, ?)",
        (text, vec.tobytes(), top["label"], float(top["score"]))
    )


def add_text_lines(lines: list[str]):
    con = _connect()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        _insert_text(con, line)
    con.commit()
    con.close()
