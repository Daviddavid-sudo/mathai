from django.shortcuts import render, redirect
from GrothendieckAi.forms import MathQuestionForm
from GrothendieckAi.utils.query_llama import answer_question_for_pdf
from qa.models import QuestionHistory, PDFDocument
from qa.forms import PDFUploadForm
import os


def home_view(request):
    return render(request, 'home.html')


def math_qa_view(request):
    question = None
    relevant_pages = []
    selected_pdf = None
    pdfs = PDFDocument.objects.all().order_by('-uploaded_at')

    if request.method == 'POST':
        question = request.POST.get('question', '').strip()
        pdf_id = request.POST.get('pdf_id')

        if pdf_id:
            try:
                selected_pdf = PDFDocument.objects.get(id=pdf_id)
            except PDFDocument.DoesNotExist:
                selected_pdf = None

        if question and selected_pdf:
            pdf_base_name = os.path.splitext(os.path.basename(selected_pdf.file.path))[0]
            answer, pages, _ = answer_question_for_pdf(pdf_base_name, question)

            # ✅ Deduplicate and sort relevant pages
            relevant_pages = sorted(set(pages)) if pages else []

    elif request.method == 'GET':
        pdf_id = request.GET.get('pdf_id')
        if pdf_id:
            try:
                selected_pdf = PDFDocument.objects.get(id=pdf_id)
            except PDFDocument.DoesNotExist:
                selected_pdf = None

    print("Relevant pages:", relevant_pages)

    return render(request, 'question_form.html', {
        'question': question,
        'relevant_pages': relevant_pages,
        'pdfs': pdfs,
        'selected_pdf': selected_pdf,
        'pdf_url': selected_pdf.file.url if selected_pdf else '',
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
