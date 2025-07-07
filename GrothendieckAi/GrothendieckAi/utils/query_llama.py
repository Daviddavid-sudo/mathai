import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json
import re
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def load_index_and_texts_for_pdf(pdf_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Always safe
    index_path = os.path.join(base_dir, "..", "data", "pdf_chunks", f"{pdf_name}_chunks.index")
    json_path = os.path.join(base_dir, "..", "data", "pdf_chunks", f"{pdf_name}_chunks.json")
    print(f"📦 Index exists: {os.path.exists(index_path)}")
    print(f"📦 JSON exists: {os.path.exists(json_path)}")

    if not os.path.exists(index_path) or not os.path.exists(json_path):
        print(f"❌ Index or chunks files not found for PDF: {pdf_name}")
        print(f"🔍 Looked for index at: {index_path}")
        print(f"🔍 Looked for JSON at: {json_path}")
        return None, []

    index = faiss.read_index(index_path)
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def search_index(query_embedding, index, k=7):
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
            timeout=90
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
        f"You are given several chunks of text from a document. For each chunk, reply only YES or NO if it likely helps answer the question.\n"
        f"Reply in EXACTLY the following format (no extra text):\n"
        f"Chunk 0: YES\n"
        f"Chunk 1: NO\n"
        f"Chunk 2: YES\n"
        f"...\n"
        f"Do not include any other explanations or comments."
    )

    for i, chunk in enumerate(candidates):
        snippet = chunk["text"][:800].replace("\n", " ").strip()
        prompt += f"Chunk {i} (Page {chunk.get('page', '?')}): {snippet}\n\n"

    response = call_llama3_2(prompt)
    print("🤖 LLaMA Response:\n", response)

    relevant_chunks = []
    for i, chunk in enumerate(candidates):
        pattern = rf"^\s*Chunk {i}( \(Page \d+\))?:\s*(YES|NO)"
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match and match.group(2).strip().upper() == "YES":
            relevant_chunks.append(chunk)

    return relevant_chunks


def answer_question_for_pdf(pdf_name, question, base_dir=BASE_DIR):
    index, texts = load_index_and_texts_for_pdf(pdf_name)
    if index is None or not texts:
        return f"Error: Index or texts not found for PDF '{pdf_name}'.", [], []

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=7)
    candidates = [texts[i] for i in top_indices if i < len(texts)]

    # Print the 5 chunks before relevance filtering
    print("\n📝 Top 7 chunks retrieved by FAISS index:")
    for i, chunk in enumerate(candidates):
        snippet = chunk["text"][:500].replace("\n", " ").strip()
        print(f"Chunk {i} (Page {chunk.get('page', '?')}): {snippet}...")

    if not candidates:
        return "No relevant context found.", [], []

    relevant_chunks = llama_select_relevant_chunks(question, candidates)

    if not relevant_chunks:
        return "No chunks were judged relevant.", [], []

    answer = "\n\n".join([f"(Page {c['page']}):\n{c['text']}" for c in relevant_chunks])
    page_list = [c['page'] for c in relevant_chunks]
    return answer, page_list, relevant_chunks

