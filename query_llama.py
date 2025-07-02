import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import ollama
import webbrowser
import re

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Math Answer</title>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
  </style>
</head>
<body>
  <h2>Answer (rendered with MathJax):</h2>
  <div>{answer}</div>
</body>
</html>
"""

def save_math_answer_as_html(answer_text, output_file="answer.html"):
    html_content = html_template.format(answer=answer_text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved answer to {output_file}")
    webbrowser.open(output_file)

def preprocess_chunk_text(text):
    # Fix common LaTeX fragments with proper escaping
    text = re.sub(r'\bP3\b', r'$\\mathbb{P}^3$', text)
    text = re.sub(r'\bOX\b', r'$\\mathcal{O}_X$', text)
    text = re.sub(r'\bωX\b', r'$\\omega_X$', text)
    text = re.sub(r'\bωY\b', r'$\\omega_Y$', text)

    # Fix example exact sequence notation
    text = re.sub(
        r'0\s*/\s*O\(-4\)\s*/\s*O\s*/\s*OX\s*/\s*0',
        r'$0 \\to \\mathcal{O}(-4) \\to \\mathcal{O} \\to \\mathcal{O}_X \\to 0$',
        text
    )

    return text

# --- Fix and convert raw math text into proper LaTeX using LLaMA ---
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
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# --- Configuration ---
INDEX_PATH = "faiss_index.index"
TEXTS_PATH = "index_texts.txt"
EMBEDDING_MODEL = "intfloat/e5-base-v2"   # or "microsoft/mathbert"
OLLAMA_MODEL = "llama3.2"

# --- Device setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model + tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

# --- Utilities ---
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
    index = faiss.read_index(index_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return index, texts

def search_index(query_embedding, index, k=10):
    D, I = index.search(query_embedding, k)
    return I[0]

def query_ollama_llama32(context_chunks, question, model_name=OLLAMA_MODEL):
    # Preprocess chunks for LaTeX fixes before sending context
    cleaned_chunks = [preprocess_chunk_text(chunk) for chunk in context_chunks]

    context = "\n\n".join(cleaned_chunks)
    prompt = f"""You are a helpful math tutor. You must answer ONLY using the information provided below.
If the context does not contain information relevant to the question, say so clearly.
Answer using **only the following excerpts from the document**.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']

# --- Main interactive loop ---
if __name__ == "__main__":
    index, texts = load_index_and_texts()

    while True:
        question = input("\nAsk your math question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        # Embed query and search
        query_embedding = embed_query(question)
        top_indices = search_index(query_embedding, index, k=10)
        relevant_chunks = [texts[i] for i in top_indices]

        # Show context chunks (for debug)
        print("\n--- 🔍 Retrieved Context Chunks ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\nChunk {i+1}:\n{chunk[:800]}")

        # Ask LLaMA with retrieved context
        raw_answer = query_ollama_llama32(relevant_chunks, question)

        # Fix math LaTeX in the raw answer
        fixed_answer = fix_math_latex_with_llama(raw_answer, model_name=OLLAMA_MODEL)

        print("\n--- 🤖 LLaMA 3.2 Answer (fixed LaTeX) ---")
        print(fixed_answer)

        # Save to HTML and open in browser
        save_math_answer_as_html(fixed_answer)

