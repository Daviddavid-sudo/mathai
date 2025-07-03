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
    """
    Calls local LLaMA 3.2 model via Ollama CLI and returns generated text.
    Adjust the call to pass the prompt correctly.
    """
    try:
        # Option 1: pass prompt as argument (may fail if prompt is too long)
        # result = subprocess.run(
        #     ["ollama", "run", "llama3.2", prompt],
        #     capture_output=True,
        #     text=True,
        #     timeout=30
        # )
        
        # Option 2: pass prompt via stdin (recommended for long prompts)
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


def llama_select_best_chunk(question, candidates):
    print("\n🔍 Top 5 Candidates:")
    for i, c in enumerate(candidates):
        print(f"\n--- Candidate {i} (Page {c.get('page', '?')}) ---\n{c['text'][:1000]}...\n")

    prompt = f"You are a helpful assistant. A user asks: {question}\n\n"
    prompt += "Here are 5 text chunks from a document:\n"
    for i, chunk in enumerate(candidates):
        snippet = chunk['text'][:1000].replace("\n", " ").strip()
        prompt += f"\nChunk {i} (Page {chunk.get('page', '?')}): {snippet}"
    prompt += "\n\nPlease reply only with the number of the chunk that best answers the question."

    print("\n🧠 Prompt Preview:\n" + prompt[:1500] + "\n... [truncated]")

    response = call_llama3_2(prompt)
    print("🤖 LLaMA Response:", response)

    match = re.search(r"(\d+)", response)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(candidates):
            return candidates[idx]

    print("⚠️ Failed to parse a valid chunk number from LLaMA response, falling back to first candidate.")
    return candidates[0]

def answer_question_pipeline(question):
    if index is None or not texts:
        return "Error: FAISS index or texts not found.", []

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=5)
    candidates = [texts[i] for i in top_indices if i < len(texts)]

    if not candidates:
        return "No relevant context found.", []

    best_chunk = llama_select_best_chunk(question, candidates)

    answer = f"Most relevant result is from page {best_chunk['page']}:\n\n{best_chunk['text']}"
    return answer, [best_chunk['page']]

# Load index and texts at module level
index, texts = load_index_and_texts()

