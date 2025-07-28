import streamlit as st
import sqlite3
from ingest import add_pdf
from rag import explain
from cluster import build_clusters
from plot import make_timeline_new
from db import get_connection

st.set_page_config("Girlfriend Emotion‑Tracker")

if "step" not in st.session_state:
    st.session_state.step = 0

# Use shared DB connection
con = get_connection()

# Create feedback table if it doesn't exist
con.execute("""
CREATE TABLE IF NOT EXISTS answer_feedback (
    emotion TEXT,
    answer TEXT,
    res TEXT CHECK(res IN ('it_is', 'never'))
)
""")
con.commit()

# --- Page 0: Upload ---
if st.session_state.step == 0:
    st.title("Upload your chat archive")
    up = st.file_uploader("PDFs or text files", accept_multiple_files=True)
    if st.button("Process") and up:
        for f in up:
            add_pdf(f)
        build_clusters()
        st.session_state.step = 1
        st.rerun()

# --- Page 1: Emotion Insights ---
else:
    st.header("How is she feeling?")
    emo = st.selectbox(
        "Pick an emotion",
        ["angry", "happy", "sad", "scared", "surprise", "disgust"]
    )
    answer_ph = st.empty()

    if st.button("Explain"):
        result, _ = explain(emo)
        answer_ph.write(result)

        with st.expander("Rate this explanation"):
            col1, col2 = st.columns([3, 3])
            if col1.button("✅ It is this"):
                con.execute(
                    "INSERT INTO answer_feedback (emotion, answer, res) VALUES (?, ?, ?)",
                    (emo, result, "it_is")
                )
                con.commit()
                st.success("Marked as helpful ✅")

            if col2.button("❌ nope"):
                con.execute(
                    "INSERT INTO answer_feedback (emotion, answer, res) VALUES (?, ?, ?)",
                    (emo, result, "never")
                )
                con.commit()
                st.warning("Marked as unhelpful ❌")

    st.divider()
    colA, colB = st.columns(2)

    if colA.button("Fine‑tune on feedback"):
        st.info("Training disabled — this project no longer uses fine-tuning feedback.")

    if colB.button("Show top triggers"):
        png = make_timeline_new()
        st.image(png)

    st.divider()
    st.caption(
        "All data & models live only inside this folder. Nothing is sent to the cloud."
    )
