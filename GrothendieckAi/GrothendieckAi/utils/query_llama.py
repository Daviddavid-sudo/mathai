import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(BASE_DIR, "index_chunks.json")  # JSON file with chunks and pages

EMBEDDING_MODEL = "intfloat/e5-base-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH):
    try:
        index = faiss.read_index(index_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            texts_with_pages = json.load(f)  # list of dicts {"page": int, "text": str}
        return index, texts_with_pages
    except Exception as e:
        print(f"Error loading index/texts: {e}")
        return None, []

print("Loading embedding model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_query(query):
    encoded = tokenizer(f"query: {query}", return_tensors="pt", truncation=True, padding=True, max_length=512)
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        model_output = model(**encoded)
    embedding = mean_pooling(model_output, encoded["attention_mask"])
    return embedding.cpu().numpy()

def search_index(query_embedding, index, k=1):  # Only get top 1
    if index is None:
        return []
    D, I = index.search(query_embedding, k)
    return I[0]

def answer_question_pipeline(question):
    if index is None or not texts:
        return "Error: FAISS index or texts not found.", []

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=1)  # Only top 1
    if not top_indices.size or top_indices[0] >= len(texts):
        return "No relevant context found in the index.", []

    best_chunk = texts[top_indices[0]]

    # Prepare answer showing the best chunk and its page
    answer = f"Most relevant result is from page {best_chunk['page']}:\n\n{best_chunk['text']}"

    return answer, [best_chunk['page']]

# Load index and texts once on import for global use by all functions
index, texts = load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH)

