from django.shortcuts import render
from .forms import MathQuestionForm
from .utils.query_llama import (
    load_index_and_texts,
    query_ollama_with_context_and_question,
    search_index,
    embed_query,
    INDEX_PATH,
    TEXTS_PATH
)
import os

OLLAMA_MODEL = "llama3.2"

# Load index and texts once at module load
index, texts = load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH)

def math_qa_view(request):
    print("🔥 Request method:", request.method)

    question = None
    answer = None
    retrieved_chunks = []

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        print("🔥 Question received:", repr(question))
        print("🔥 Index is None?", index is None)
        print("🔥 Texts loaded?", bool(texts))

        if question and index is not None and texts:
            query_embedding = embed_query(question)
            print("✅ Embedded query.")

            top_indices = search_index(query_embedding, index, k=10)
            print("✅ Searched index. Top indices:", top_indices)

            retrieved_chunks = [texts[i] for i in top_indices if i < len(texts)]
            print(f"✅ Retrieved {len(retrieved_chunks)} chunks.")

            if retrieved_chunks:
                print("\n--- Retrieved Chunks ---")
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    print(f"Chunk {i}: {chunk[:500]}...")  # first 500 chars
                raw_answer = query_ollama_with_context_and_question(retrieved_chunks, question)
                print("\n--- Raw Answer ---")
                print(raw_answer)
                answer = raw_answer
            else:
                answer = "No relevant context found in the index."

        else:
            answer = "Error loading index or no question provided."

    print("✅ Rendering template.")
    print(f"✅ Rendering template with question: {question}, answer: {answer}, chunks count: {len(retrieved_chunks)}")

    return render(request, 'question_form.html', {
        'question': question,
        'answer': answer,
        'retrieved_chunks': retrieved_chunks,
    })
