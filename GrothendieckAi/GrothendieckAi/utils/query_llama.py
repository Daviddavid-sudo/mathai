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

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

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
            texts = [line.strip() for line in f if line.strip()]
        return index, texts
    except Exception as e:
        return None, []

def search_index(query_embedding, index, k=10):
    if index is None:
        return []
    D, I = index.search(query_embedding, k)
    return I[0]

def truncate_context(text, max_length=3000):
    return text if len(text) <= max_length else text[:max_length] + " ... (truncated)"

def clean_chunks_with_llama(raw_chunks, model_name="llama3.2"):
    combined_text = "\n\n---\n\n".join(raw_chunks)
    prompt = f"""
You are a LaTeX expert assistant. Convert the following raw math text into clean, properly formatted LaTeX.
- Use single backslashes.
- Wrap math expressions in appropriate math delimiters ($...$ for inline, \\[ ... \\] for display).
- Do not add explanations or commentary, output only the LaTeX.

Text to convert:
{combined_text}

Output:
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception:
        return combined_text

def fix_math_latex_with_llama(raw_text, model_name="llama3.2"):
    prompt = f"""
You are a LaTeX expert and helpful assistant. Convert the following math text to clean, properly formatted LaTeX.

Requirements:
- Use single backslash `\\` for LaTeX commands (e.g. \\mathcal{{O}})
- Wrap all math expressions in appropriate math delimiters: `$...$` for inline math, `\\[ ... \\]` for display math.
- Use `\\text{{...}}` only inside math mode for text fragments.
- Fix any escaping issues or misplaced backslashes.
- Avoid double backslashes before LaTeX commands.
- Do not include extra explanations, output only the corrected LaTeX.

Here is the text to convert:

{raw_text}

Output:
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception:
        return raw_text

def query_ollama_llama32(context_chunks, question, model_name=OLLAMA_MODEL, return_chunks=False):
    cleaned_context = clean_chunks_with_llama(context_chunks, model_name=model_name)
    truncated_context = truncate_context(cleaned_context, max_length=3000)

    if not truncated_context.strip():
        return ("The context does not contain any relevant information to answer the question.",
                context_chunks if return_chunks else None)

    prompt = f"""You are a helpful math tutor. You must answer ONLY using the information provided below.
If the context does not contain information relevant to the question, say so clearly.

Context:
{truncated_context}

Question: {question}
Answer:"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response['message']['content']
    except Exception:
        answer = "Sorry, I couldn't generate an answer due to a model error."

    return (answer, context_chunks if return_chunks else None)

def answer_question_pipeline(question, top_k=10):
    index, texts = load_index_and_texts()
    if index is None or not texts:
        return "Error: FAISS index or texts not found."

    query_embedding = embed_query(question)
    top_indices = search_index(query_embedding, index, k=top_k)
    relevant_chunks = [texts[i] for i in top_indices if i < len(texts)]

    if not relevant_chunks:
        return "No relevant context found in the index."

    raw_answer, _ = query_ollama_llama32(relevant_chunks, question)
    fixed_answer = fix_math_latex_with_llama(raw_answer, model_name=OLLAMA_MODEL)

    return fixed_answer


