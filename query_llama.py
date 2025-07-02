from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

INDEX_PATH = "faiss_index.index"
TEXTS_PATH = "index_texts.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # You can swap this for better ones later
OLLAMA_MODEL = "llama3.2"                # Make sure you ran `ollama pull llama3`

def load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH):
    index = faiss.read_index(index_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return index, texts

def embed_query(query, model):
    return model.encode([query], convert_to_numpy=True)

def search_index(query_embedding, index, k=20):
    D, I = index.search(query_embedding, k)
    return I[0]

def query_ollama_llama32(context_chunks, question, model_name=OLLAMA_MODEL):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful math tutor. You must answer ONLY using the information provided below. 
    Answer using **only the following excerpts from the document."

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']

if __name__ == "__main__":
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    index, texts = load_index_and_texts()

    while True:
        question = input("\nAsk your math question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        # Embed the query and retrieve top chunks
        query_embedding = embed_query(question, embedder)
        top_indices = search_index(query_embedding, index, k=20)
        relevant_chunks = [texts[i] for i in top_indices]

        # DEBUG: print retrieved context
        print("\n--- 🔍 Retrieved Context Chunks ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\nChunk {i+1}:\n{chunk[:800]}")  # Show only first 800 chars

        # Query Ollama with retrieved context
        answer = query_ollama_llama32(relevant_chunks, question)

        print("\n--- 🤖 LLaMA 3.2 Answer (via Ollama) ---")
        print(answer)
