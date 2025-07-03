import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# Device & model initialization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

# Load chunks from your file; expects --- Page <num> --- separator in file
def load_chunks(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    entries = text.split("--- Page ")
    chunks = []
    for entry in entries[1:]:
        page_str, content = entry.split("---", 1)
        chunks.append({
            "page": int(page_str.strip()),
            "text": content.strip()
        })
    return chunks

# Mean pooling function for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # last hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# Embed chunk texts in batches
def embed_chunks(chunks, batch_size=32):
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_prefixed = [f"passage: {text}" for text in batch_texts]

        inputs = tokenizer(batch_prefixed, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs['attention_mask'])

        all_embeddings.append(embeddings.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return final_embeddings

# Build FAISS index for embeddings
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Save FAISS index to file
def save_index(index, path="GrothendieckAi/GrothendieckAi/utils/faiss_index.index"):
    faiss.write_index(index, path)

# Save chunks with page info as JSON
def save_chunks_with_pages(chunks, path="GrothendieckAi/GrothendieckAi/utils/index_chunks.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    txt_path = "GrothendieckAi/GrothendieckAi/utils/output_chunks.txt"  # Your cleaned input file with page markers
    print("🔄 Loading chunks...")
    chunks = load_chunks(txt_path)
    print(f"📄 Loaded {len(chunks)} chunks with page info.")

    print("🔄 Embedding chunks...")
    embeddings = embed_chunks(chunks)
    print(f"✅ Created embeddings with shape {embeddings.shape}.")

    print("🔄 Building FAISS index...")
    index = build_faiss_index(embeddings)
    print("✅ Index built.")

    print("💾 Saving index and chunks...")
    save_index(index)
    save_chunks_with_pages(chunks)
    print("✅ Index and chunks saved.")
