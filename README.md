some notes when u set it up its a bit slow cuz its offline and keeps ur data secure
when u upload the pdf just click it once it takes a second don't double click the file will crash
💔 Girlfriend Emotion‑Tracker
Ever wish you could figure out why she’s mad this time? This tool helps you analyze your chat history to detect emotional patterns, explain her feelings, and gradually adapt to your unique relationship dynamic.

📦 Setup Instructions
Note: This repo does not include model weights due to GitHub file size limits.

To run the app locally:

1. Clone the repo
bash
Copy code
git clone https://github.com/yourusername/girlfriend-emotion-tracker.git
cd girlfriend-emotion-tracker
2. Create a virtual environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
3. Download model files manually
Sentence embedding model (384-dim)
Download all-MiniLM-L6-v2 from HuggingFace:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Place it in: models/embedding/

Quantized Mistral 7B model
Download mistral-7b-instruct-v0.2.Q4_K_M.gguf from TheBloke's HuggingFace repo:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
Place it in: models/llm/

4. Run the app
bash
Copy code
streamlit run main.py
🧠 How It Works
On the surface, it’s a button that tells you why she’s upset. But under the hood:

Message Embeddings
Every message gets turned into a 384‑dimensional vector — think GPS coordinates in "meaning‑land" — using a MiniLM encoder.

Fast Semantic Search
A FAISS HNSW index lets us retrieve the top 10 closest messages to any emotion-based query in milliseconds.

Emotion Detection
We use a GoEmotions classifier to assign 27 possible emotional labels to every message in your history.

LLM Explanation
These messages and their emotional tags are passed to a quantized 7‑billion‑parameter Mistral model — running entirely locally, so there are no cloud fees or data leaks.


🙋 Why?
Because relationships are hard, and we might as well use AI to give us a fighting chance.
Its a bit slow cuz the models are big and offline
