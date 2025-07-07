from django.shortcuts import render, redirect
from GrothendieckAi.forms import MathQuestionForm
from GrothendieckAi.utils.query_llama import answer_question_for_pdf
from qa.models import QuestionHistory, PDFDocument
from qa.forms import PDFUploadForm, QuestionHistoryForm
import os
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

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
            relevant_pages = sorted(set(pages)) if pages else []

            # Save question history
            if request.user.is_authenticated:
                QuestionHistory.objects.create(
                    user=request.user,
                    question=question,
                    pdf=selected_pdf,         # if using ForeignKey
                    source_pages=relevant_pages
        )

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


def history_view(request):
    if request.user.is_authenticated:
        # Assuming you have a model to track Q&A history, e.g. QuestionHistory
        history = QuestionHistory.objects.filter(user=request.user).order_by('-timestamp')
    else:
        history = []
    return render(request, 'history.html', {'history': history})


@login_required
def edit_question_history(request, pk):
    entry = get_object_or_404(QuestionHistory, pk=pk, user=request.user)

    if request.method == 'POST':
        form = QuestionHistoryForm(request.POST, instance=entry)
        if form.is_valid():
            form.save()
            return redirect('history')
    else:
        form = QuestionHistoryForm(instance=entry)

    return render(request, 'edit_history.html', {'form': form, 'entry': entry})


@login_required
def delete_question_history(request, pk):
    entry = get_object_or_404(QuestionHistory, pk=pk, user=request.user)

    if request.method == 'POST':
        entry.delete()
        return redirect('history')

    return render(request, 'delete_history.html', {'entry': entry})