from django.shortcuts import render
from .forms import MathQuestionForm
from .utils.query_llama import (
    load_index_and_texts,
    search_index,
    embed_query,
    INDEX_PATH,
    TEXTS_PATH
)

# Load index and texts once at module load
index, texts = load_index_and_texts(index_path=INDEX_PATH, texts_path=TEXTS_PATH)

def math_qa_view(request):
    print("🔥 Request method:", request.method)

    question = None
    answer = None
    retrieved_chunk = None
    relevant_page = None

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        print("🔥 Question received:", repr(question))
        print("🔥 Index is None?", index is None)
        print("🔥 Texts loaded?", bool(texts))

        if question and index is not None and texts:
            query_embedding = embed_query(question)
            top_indices = search_index(query_embedding, index, k=1)  # only top 1
            
            if top_indices.size and top_indices[0] < len(texts):
                retrieved_chunk = texts[top_indices[0]]
                relevant_page = retrieved_chunk.get('page')
                answer = f"Most relevant result is from page {relevant_page}:\n\n{retrieved_chunk.get('text', '')}"
            else:
                answer = "No relevant context found in the index."
                relevant_page = None

    print("✅ Rendering template.")
    print(f"✅ Rendering template with question: {question}, answer: {answer}")

    return render(request, 'question_form.html', {
        'question': question,
        'relevant_page': relevant_page,
        'pdf_url': '/media/K3Global.pdf',
    })
