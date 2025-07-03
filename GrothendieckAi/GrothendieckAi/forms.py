from django import forms

class MathQuestionForm(forms.Form):
    question = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4, "cols": 50, "placeholder": "Type your math question here..."}),
        label="Your Math Question",
        max_length=1000,
    )
