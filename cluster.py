import numpy as np
import pickle
import sqlite3
from pathlib import Path
from sklearn.cluster import KMeans

DB = Path("girlfriend.db")
MODEL = Path("cluster.pkl")


def build_clusters(k: int = 8):
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT id,embedding FROM msgs").fetchall()
    con.close()

    # 1) EMPTY‐DB guard
    if not rows:
        print("⚠️  No embeddings found — run ingest first.")
        return

    ids, vecs = zip(*[
        (r[0], np.frombuffer(r[1], dtype='float32'))
        for r in rows
    ])

    # 2) TOO‐FEW‐POINTS guard
    if len(vecs) < k:
        k = len(vecs)
        print(f"⚠️  Fewer points ({len(vecs)}) than clusters — reducing k to {k}")

    mat = np.vstack(vecs)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(mat)

    # 3) ATOMIC WRITE (avoid partial writes on crash)
    with open(MODEL, "wb") as fp:
        pickle.dump({"model": kmeans, "ids": ids}, fp)


def label_for_text(text_id: int):
    if not MODEL.exists():
        raise FileNotFoundError(f"{MODEL} not found — run build_clusters() first.")
    cfg = pickle.load(open(MODEL, "rb"))
    ids = cfg["ids"]
    if text_id not in ids:
        raise ValueError(f"Text ID {text_id} not in cluster model.")
    km = cfg["model"]
    idx = ids.index(text_id)
    return int(km.labels_[idx])
