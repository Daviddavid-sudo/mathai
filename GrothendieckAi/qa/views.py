from django.shortcuts import render, redirect
from GrothendieckAi.forms import MathQuestionForm
from GrothendieckAi.utils.query_llama import answer_question_pipeline
from qa.models import QuestionHistory, PDFDocument
from qa.forms import PDFUploadForm


def home_view(request):
    return render(request, 'home.html')


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


def history_view(request):
    history = QuestionHistory.objects.order_by('-timestamp')[:50]
    return render(request, 'history.html', {'history': history})


def library_view(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('library')  # Redirect to avoid re-posting on refresh
    else:
        form = PDFUploadForm()

    pdfs = PDFDocument.objects.all().order_by('-uploaded_at')
    return render(request, 'library.html', {'form': form, 'pdfs': pdfs})
