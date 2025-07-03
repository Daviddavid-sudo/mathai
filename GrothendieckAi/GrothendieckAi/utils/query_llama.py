import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import ollama
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
TEXTS_PATH = os.path.join(BASE_DIR, "index_texts.txt")

EMBEDDING_MODEL = "intfloat/e5-base-v2"
OLLAMA_MODEL = "llama3.2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH):
    try:
        index = faiss.read_index(index_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return index, texts
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

def search_index(query_embedding, index, k=10):
    if index is None:
        return []
    D, I = index.search(query_embedding, k)
    return I[0]

def truncate_context(text, max_length=3000):
    return text if len(text) <= max_length else text[:max_length] + " ... (truncated)"

def query_ollama_with_context_and_question(context_chunks, question, model_name=OLLAMA_MODEL):
    combined_context = "\n\n---\n\n".join(context_chunks)
    truncated_context = truncate_context(combined_context, max_length=3000)

    if not truncated_context.strip():
        return "The context does not contain any relevant information to answer the question."

    prompt = f"""
You are a LaTeX expert math tutor. Use only the information provided below to answer the question.

📌 Format math as follows:
- Use `$...$` for inline math.
- Use `\\[ ... \\]` for display math (like equations).
- Only use `\\text{{...}}` inside math mode.
- Do not wrap entire sentences in `\\text{{}}`.

Context:
{truncated_context}

Question: {question}
Answer:
"""


    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama API error: {e}")
        return "Sorry, I couldn't generate an answer due to a model error."

def answer_question_pipeline(question, top_k=10):
    if index is None or not texts:
        return "Error: FAISS index or texts not found."

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=top_k)
    relevant_chunks = [texts[i] for i in top_indices if i < len(texts)]

    if not relevant_chunks:
        return "No relevant context found in the index."

    answer = query_ollama_with_context_and_question(relevant_chunks, question)
    return answer

# Load index and texts once on import for global use by all functions
index, texts = load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH)
