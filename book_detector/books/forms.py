from django import forms
from django.contrib.auth.models import User
from .models import Detection


class DetectionUploadForm(forms.ModelForm):
    class Meta:
        model = Detection
        fields = ['uploaded_image', 'title']  # Додано поле 'title'
        widgets = {
            'title': forms.TextInput(attrs={
                'placeholder': 'Enter detection title',
                'class': 'form-control',
            }),
            'uploaded_image': forms.ClearableFileInput(attrs={
                'class': 'form-control',
            }),
        }


class SaveBooksForm(forms.Form):
    """
    Form for saving detected books to UniqueBook.
    """
    books = forms.CharField(widget=forms.HiddenInput(), required=False)  # JSON for detected books


class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
