import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json
import re
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(BASE_DIR, "index_chunks.json")

EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model and tokenizer once
print("🔄 Loading embedding model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_query(query):
    encoded = tokenizer(f"query: {query}", return_tensors="pt", truncation=True, padding=True, max_length=512)
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        model_output = model(**encoded)
    embedding = mean_pooling(model_output, encoded["attention_mask"])
    return embedding.cpu().numpy()

def load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH):
    try:
        index = faiss.read_index(index_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            texts_with_pages = json.load(f)
        return index, texts_with_pages
    except Exception as e:
        print(f"❌ Error loading index/texts: {e}")
        return None, []

def search_index(query_embedding, index, k=5):
    if index is None:
        return []
    D, I = index.search(query_embedding, k)
    return I[0]

def call_llama3_2(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"❌ Ollama CLI error: {result.stderr}")
            return ""
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Exception running llama3.2 CLI: {e}")
        return ""

def llama_select_relevant_chunks(question, candidates):
    prompt = (
        f"You are a helpful AI assistant. A user asks the question:\n\n"
        f"\"{question}\"\n\n"
        f"You are given several chunks of text from a document. For each chunk, reply YES if it likely helps answer the question, otherwise reply NO.\n\n"
        f"Respond using the following format:\n"
        f"Chunk 0: YES\nChunk 1: NO\n...\n\n"
    )

    for i, chunk in enumerate(candidates):
        snippet = chunk["text"][:800].replace("\n", " ").strip()
        prompt += f"Chunk {i} (Page {chunk.get('page', '?')}): {snippet}\n\n"

    response = call_llama3_2(prompt)
    print("🤖 LLaMA Response:\n", response)

    relevant_chunks = []
    for i, chunk in enumerate(candidates):
        # Updated regex to handle optional "(Page N)" and optional leading spaces
        pattern = rf"^\s*Chunk {i}( \(Page \d+\))?:\s*(YES|NO)"
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match and match.group(2).strip().upper() == "YES":
            relevant_chunks.append(chunk)

    return relevant_chunks

def answer_question_pipeline(question):
    if index is None or not texts:
        return "Error: FAISS index or texts not found.", [], []

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=5)
    candidates = [texts[i] for i in top_indices if i < len(texts)]

    if not candidates:
        return "No relevant context found.", [], []

    relevant_chunks = llama_select_relevant_chunks(question, candidates)

    if not relevant_chunks:
        return "No chunks were judged relevant.", [], []

    answer = "\n\n".join([f"(Page {c['page']}):\n{c['text']}" for c in relevant_chunks])
    page_list = [c['page'] for c in relevant_chunks]
    return answer, page_list, relevant_chunks

# Load index and texts at module level
index, texts = load_index_and_texts()
