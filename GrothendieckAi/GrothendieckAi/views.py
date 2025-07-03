from django.shortcuts import render
from .forms import MathQuestionForm
from .utils.query_llama import answer_question_pipeline

def math_qa_view(request):
    question = None
    relevant_pages = []

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        if question:
            _, pages, _ = answer_question_pipeline(question)
            relevant_pages = pages if pages else []

    return render(request, 'question_form.html', {
        'question': question,
        'relevant_pages': relevant_pages,
        'pdf_url': '/media/K3Global.pdf',
    })

