from django import forms
from .models import PDFDocument, QuestionHistory

class QuestionHistoryForm(forms.ModelForm):
    class Meta:
        model = QuestionHistory
        fields = ['question', 'source_pages']  # Add other editable fields as needed
        widgets = {
            'question': forms.Textarea(attrs={'rows': 3}),
            'source_pages': forms.Textarea(attrs={'rows': 3}),  # JSON editing, you can customize
        }

class PDFUploadForm(forms.ModelForm):
    class Meta:
        model = PDFDocument
        fields = ['title', 'file']
