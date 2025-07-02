from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your chunks from txt
def load_chunks(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    entries = text.split("--- Page ")
    chunks = []
    for entry in entries[1:]:
        page, content = entry.split("---", 1)
        chunks.append({
            "page": int(page.strip()),
            "text": content.strip()
        })
    return chunks

# Embed chunks
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, texts

# Create FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Save FAISS index
def save_index(index, path="faiss_index.index"):
    faiss.write_index(index, path)

# Save the raw texts to align with index
def save_texts(texts, path="index_texts.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n\n")

if __name__ == "__main__":
    txt_path = "output_chunks.txt"  # Your cleaned file
    chunks = load_chunks(txt_path)

    embeddings, texts = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))

    save_index(index)
    save_texts(texts)

    print(f"✅ Indexed {len(texts)} chunks.")
