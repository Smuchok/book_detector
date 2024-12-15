# Generated by Django 5.1.3 on 2024-12-07 02:00

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("books", "0009_uniquebook_user"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="UserAction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "action_type",
                    models.CharField(
                        choices=[
                            ("profile_update", "Profile Update"),
                            ("add_detection", "Add Detection"),
                            ("add_unique_books", "Add Unique Books"),
                            ("register", "Register"),
                            ("login", "Login"),
                            ("logout", "Logout"),
                            ("other", "Other"),
                        ],
                        max_length=50,
                    ),
                ),
                ("description", models.TextField(blank=True, null=True)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                (
                    "detection",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="actions",
                        to="books.detection",
                    ),
                ),
                (
                    "unique_books",
                    models.ManyToManyField(
                        blank=True, related_name="actions", to="books.uniquebook"
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="actions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]