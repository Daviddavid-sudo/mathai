from django.db import models
from django.contrib.auth.models import User

class QuestionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    question = models.TextField()
    answer = models.TextField()
    source_pages = models.JSONField(null=True, blank=True)  # Optional: list of pages or any extra info
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Question at {self.timestamp}: {self.question[:50]}..."



class PDFDocument(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
