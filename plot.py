import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import uuid


def make_timeline_new():
    con = sqlite3.connect("girlfriend.db")
    rows = con.execute("""
        SELECT emotion, text, MAX(score)
        FROM msgs
        GROUP BY emotion
    """).fetchall()
    con.close()

    if not rows:
        raise ValueError("No emotion data found.")

    df = pd.DataFrame(rows, columns=["emotion", "text", "score"])
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    if df.empty:
        raise ValueError("No valid emotion scores found.")

    y_labels = [
        f"{t[:30]}... ({emo})"
        for t, emo in zip(df["text"], df["emotion"])
    ]

    height = max(4.0, len(y_labels) * 0.5)
    fig, ax = plt.subplots(figsize=(8.0, height))
    ax.barh(y_labels, df["score"], color="#4B9BFF")  # blue!
    ax.invert_yaxis()
    ax.set_xlabel("Emotion Intensity")
    ax.set_title("Top Message per Emotion")
    plt.tight_layout()

    fname = f"emotion_summary_{uuid.uuid4().hex[:6]}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
