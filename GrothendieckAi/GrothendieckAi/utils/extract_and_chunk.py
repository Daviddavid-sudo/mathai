import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Config ---
TEXT_PATH = "output_chunks.txt"               # The .txt version of your PDF
INDEX_PATH = "faiss_index.index"
TEXTS_OUT_PATH = "index_texts.txt"
CHUNK_SIZE = 1000                     # Can tune this
CHUNK_OVERLAP = 200                   # For continuity between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"      # Can swap later

# --- Load and clean text ---
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return clean_text(text)

def clean_text(text):
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Fix common ligatures
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    # Remove lines with just numbers (likely page numbers)
    text = re.sub(r'\b\d+\b', '', text)
    # Remove "Contents" or chapter headings
    text = re.sub(r'(?i)\b(contents|chapter \d+|table of contents)\b', '', text)
    return text.strip()


# --- Chunking ---
def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # Split by sentences and group into chunks
    sentences = re.split(r'(?<=[\.\?!])\s+(?=[A-Z\\])', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap
    final_chunks = []
    for i in range(0, len(chunks)):
        start = max(i - 1, 0)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined.strip())

    return final_chunks

# --- Embedding & FAISS ---
def embed_chunks(chunks, model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, chunks

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path=INDEX_PATH):
    faiss.write_index(index, path)

def save_chunks(chunks, path=TEXTS_OUT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")

# --- Main ---
if __name__ == "__main__":
    print("📄 Loading text...")
    raw_text = load_text(TEXT_PATH)

    print("🔪 Chunking...")
    chunks = split_into_chunks(raw_text)

    print(f"📚 Created {len(chunks)} chunks")

    print("🧠 Embedding...")
    embeddings, cleaned_chunks = embed_chunks(chunks)

    print("💾 Saving FAISS index...")
    index = build_faiss_index(np.array(embeddings))
    save_index(index)

    print("📝 Saving text chunks...")
    save_chunks(cleaned_chunks)

    print("✅ Done!")
