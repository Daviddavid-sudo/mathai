from django.shortcuts import render
from .forms import MathQuestionForm
from .utils.query_llama import answer_question_pipeline

def math_qa_view(request):
    print("🔥 Request method:", request.method)

    question = "What is the definition of a K3 surface"
    answer = None
    relevant_page = None
    retrieved_chunk = None

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        print("🔥 Question received:", repr(question))

        if question:
            # ⛏️ Embed + search + rerank
            answer, page_list, retrieved_chunk = answer_question_pipeline(question)

            if page_list:
                relevant_page = page_list[0]

    print("✅ Rendering template.")
    print(f"✅ Question: {question}")
    print(f"✅ Answer: {answer}")
    print(f"✅ Page: {relevant_page}")
    print(f"✅ Chunk: {retrieved_chunk[:100] if retrieved_chunk else 'None'}")

    return render(request, 'question_form.html', {
        'question': question,
        'answer': answer,
        'relevant_page': relevant_page,
        'retrieved_chunk': retrieved_chunk,
        'pdf_url': '/media/K3Global.pdf',
    })


