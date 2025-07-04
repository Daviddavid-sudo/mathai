import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import glob
import sys

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

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

def clean_page_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r'(?i)\b(contents|chapter \d+|table of contents)\b', '', text)
    return text.strip()

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

def embed_chunks(chunks, model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path):
    faiss.write_index(index, path)

def save_chunks_json(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

def get_safe_filename(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext

def process_pdf_chunks(raw_text_path, output_dir="GrothendieckAi/GrothendieckAi/utils/"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Loading raw text from {raw_text_path}...")
    raw_text = load_raw_text(raw_text_path)

    print("🔪 Splitting and chunking with page info...")
    chunks_with_pages = process_text_into_chunks_with_pages(raw_text)
    print(f"📝 Created {len(chunks_with_pages)} chunks.")

    print("🧠 Embedding chunks...")
    embeddings = embed_chunks(chunks_with_pages)

    print("💾 Building FAISS index...")
    index = build_faiss_index(np.array(embeddings))

    base_name = get_safe_filename(raw_text_path)
    index_path = os.path.join(output_dir, f"{base_name}.index")
    chunks_json_path = os.path.join(output_dir, f"{base_name}_chunks.json")

    print(f"💾 Saving index to {index_path} and chunks JSON to {chunks_json_path}...")
    save_index(index, index_path)
    save_chunks_json(chunks_with_pages, chunks_json_path)

    print("✅ Done with", raw_text_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_and_chunk.py <path_to_txt_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in folder {folder_path}")
        sys.exit(1)

    print(f"Found {len(txt_files)} .txt files in {folder_path}")
    for txt_file in txt_files:
        process_pdf_chunks(txt_file)
