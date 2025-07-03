import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Config ---
RAW_TEXT_PATH = "GrothendieckAi/GrothendieckAi/utils/output_chunks.txt"
INDEX_PATH = "GrothendieckAi/GrothendieckAi/utils/faiss_index.index"
CHUNKS_JSON_PATH = "GrothendieckAi/GrothendieckAi/utils/index_chunks.json"


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Load raw text ---
def load_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- Split into pages ---
def split_into_pages(raw_text):
    entries = raw_text.split('--- Page ')
    pages = []
    for entry in entries[1:]:
        page_num_str, content = entry.split('---', 1)
        page_num = int(page_num_str.strip())
        pages.append({
            "page": page_num,
            "text": content.strip()
        })
    return pages

# --- Clean page text ---
def clean_page_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r'(?i)\b(contents|chapter \d+|table of contents)\b', '', text)
    return text.strip()

# --- Chunk text with overlap ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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

    # Add overlap between chunks
    final_chunks = []
    for i in range(len(chunks)):
        start = max(i - 1, 0)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined.strip())

    return final_chunks

# --- Process raw text into chunks with page numbers ---
def process_text_into_chunks_with_pages(raw_text):
    pages = split_into_pages(raw_text)
    all_chunks = []

    for page in pages:
        cleaned_text = clean_page_text(page['text'])
        page_chunks = chunk_text(cleaned_text)

        for chunk in page_chunks:
            all_chunks.append({
                "page": page["page"],
                "text": chunk
            })

    return all_chunks

# --- Embed chunks ---
def embed_chunks(chunks, model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

# --- Build FAISS index ---
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Save index ---
def save_index(index, path=INDEX_PATH):
    faiss.write_index(index, path)

# --- Save chunks with page info ---
def save_chunks_json(chunks, path=CHUNKS_JSON_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

# --- Main ---
if __name__ == "__main__":
    print("📄 Loading raw text...")
    raw_text = load_raw_text(RAW_TEXT_PATH)

    print("🔪 Splitting and chunking with page info...")
    chunks_with_pages = process_text_into_chunks_with_pages(raw_text)
    print(f"📝 Created {len(chunks_with_pages)} chunks.")

    print("🧠 Embedding chunks...")
    embeddings = embed_chunks(chunks_with_pages)

    print("💾 Building FAISS index...")
    index = build_faiss_index(np.array(embeddings))

    print("💾 Saving FAISS index...")
    save_index(index)

    print("💾 Saving chunks with page info...")
    save_chunks_json(chunks_with_pages)

    print("✅ All done!")
