from django.shortcuts import render
from .forms import MathQuestionForm
from .utils.query_llama import answer_question_pipeline

def math_qa_view(request):
    print("🔥 Request method:", request.method)

    question = "What is the definition of a K3 surface?"
    answer = None
    relevant_page = None
    retrieved_chunk = None

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        print("🔥 Question received:", repr(question))

        if question:
            # answer_question_pipeline returns answer text and list of relevant pages
            answer, page_list = answer_question_pipeline(question)

            if page_list:
                relevant_page = page_list[0]

            # For demonstration, let's set retrieved_chunk as the answer itself
            # You can modify answer_question_pipeline to also return the chunk text separately if needed
            retrieved_chunk = answer

    print("✅ Rendering template.")
    print(f"✅ Question: {question}")
    print(f"✅ Answer: {answer}")
    print(f"✅ Relevant Page: {relevant_page}")

    return render(request, 'question_form.html', {
        'question': question,
        'answer': answer,
        'relevant_page': relevant_page,
        'retrieved_chunk': retrieved_chunk,
        'pdf_url': '/media/K3Global.pdf',
    })

