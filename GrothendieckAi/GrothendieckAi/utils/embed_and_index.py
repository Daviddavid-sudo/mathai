from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize the model + tokenizer once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "intfloat/e5-base-v2"  # Or "microsoft/mathbert"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is last hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_chunks(chunks, batch_size=32):
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []
    all_texts = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_prefixed = [f"passage: {text}" for text in batch_texts]

        inputs = tokenizer(batch_prefixed, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs['attention_mask'])

        all_embeddings.append(embeddings.cpu())
        all_texts.extend(batch_texts)

    final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return final_embeddings, all_texts


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
