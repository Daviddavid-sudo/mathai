import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()
# === Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load embedding model ===
print("🔄 Loading embedding model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)
model.eval()

# === Load Groq API Key ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable")

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# === Embedding Helpers ===
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

# === File I/O ===
def load_index_and_texts_for_pdf(pdf_name):
    index_path = os.path.join(BASE_DIR, "..", "data", "pdf_chunks", f"{pdf_name}_chunks.index")
    json_path = os.path.join(BASE_DIR, "..", "data", "pdf_chunks", f"{pdf_name}_chunks.json")
    print(f"📦 Index exists: {os.path.exists(index_path)}")
    print(f"📦 JSON exists: {os.path.exists(json_path)}")

    if not os.path.exists(index_path) or not os.path.exists(json_path):
        print(f"❌ Index or chunks files not found for PDF: {pdf_name}")
        return None, []

    index = faiss.read_index(index_path)
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks

# === FAISS Search ===
def search_index(query_embedding, index, k=10):
    if index is None:
        return []
    D, I = index.search(query_embedding, k)
    return I[0]

# === Call Groq API ===
def call_groq_llama(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        content = res.json()
        return content["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return "Error querying Groq model."

# === Relevance Filtering ===
def llama_select_relevant_chunks(question, candidates):
    prompt = (
        f"You are a helpful AI assistant. A user asks the question:\n\n"
        f"\"{question}\"\n\n"
        f"You are given several chunks of text from a document. For each chunk, reply only YES or NO if it contains information that answers the question.\n"
        f"Reply in EXACTLY the following format (no extra text):\n"
        f"Chunk 0: YES\n"
        f"Chunk 1: NO\n"
        f"Chunk 2: YES\n"
        f"...\n"
        f"Do not include any other explanations or comments.\n"
    )

    for i, chunk in enumerate(candidates):
        snippet = chunk["text"][:800].replace("\n", " ").strip()
        prompt += f"Chunk {i} (Page {chunk.get('page', '?')}): {snippet}\n\n"

    response = call_groq_llama(prompt)  # Or call_llama3_groq if that's your function
    print("🤖 Groq LLaMA Response:\n", repr(response))  # Use repr to show line breaks and weird characters

    relevant_chunks = []
    for i, chunk in enumerate(candidates):
        # Loosen the regex to allow more flexible matches
        pattern = rf"Chunk\s*{i}.*?:\s*(YES|NO)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().upper()
            if answer == "YES":
                print(f"✅ Relevant chunk {i} added (Page {chunk.get('page', '?')})")
                relevant_chunks.append(chunk)
            else:
                print(f"❌ Chunk {i} not relevant")
        else:
            print(f"⚠️ Could not find a match for Chunk {i} in response.")

    return relevant_chunks


# === Final Answer ===
def answer_question_for_pdf(pdf_name, question, base_dir=BASE_DIR):
    index, texts = load_index_and_texts_for_pdf(pdf_name)
    if index is None or not texts:
        return f"Error: Index or texts not found for PDF '{pdf_name}'.", [], []

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=10)
    candidates = [texts[i] for i in top_indices if i < len(texts)]

    print("\n📝 Top 10 chunks retrieved by FAISS index:")
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
