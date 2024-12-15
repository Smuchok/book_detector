import os
from datetime import datetime

from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

# Create your models here.

# class User(models.Model):
#     username = models.CharField(max_length=255)
#     first_name = models.CharField(max_length=255, blank=True, null=True)
#     last_name = models.CharField(max_length=255, blank=True, null=True)
#     email = models.EmailField(max_length=255, blank=True, null=True)


def upload_to_common_dir(instance, filename):
    """
    Custom upload function to store files in a common directory separated by date.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join("common_images", current_date, filename)


class Detection(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE,
        null=True,  # Allow anonymous detections
        blank=True
    )
    title = models.CharField(max_length=255, blank=True, null=True)
    uploaded_image = models.ImageField(upload_to=upload_to_common_dir)
    detection_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Detection {self.id} by {self.user if self.user else 'Anonymous'} on {self.detection_date}"


class DetectedBook(models.Model):
    detection = models.ForeignKey(
        Detection,
        on_delete=models.CASCADE,
        related_name="detected_books"
    )
    cropped_image = models.ImageField(upload_to=upload_to_common_dir)
    rotated_image = models.ImageField(upload_to=upload_to_common_dir)
    full_text = models.TextField(blank=True, null=True)
    class_name = models.CharField(max_length=50, default="book")
    confidence = models.FloatField()
    detected_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"DetectedBook {self.id} (Confidence: {self.confidence:.2f})"


class UniqueBook(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    detection = models.ForeignKey(
        Detection,
        on_delete=models.SET_NULL,
        related_name="unique_books",
        null=True, blank=True
    )
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    cropped_image = models.ImageField(upload_to=upload_to_common_dir, blank=True, null=True)
    rotated_image = models.ImageField(upload_to=upload_to_common_dir, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} - {self.author}"


class UserAction(models.Model):
    ACTION_CHOICES = [
        ('profile_update', 'Profile Update'),
        ('add_detection', 'Add Detection'),
        ('add_unique_books', 'Add Unique Books'),
        ('register', 'Register'),
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="actions")
    action_type = models.CharField(max_length=50, choices=ACTION_CHOICES)
    description = models.TextField(blank=True, null=True)
    detection = models.ForeignKey(Detection, on_delete=models.SET_NULL, blank=True, null=True, related_name="actions")
    unique_books = models.ManyToManyField(UniqueBook, blank=True, related_name="actions")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_action_type_display()} by {self.user.username} at {self.timestamp}"
