# Generated by Django 5.1.3 on 2024-11-25 18:54

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("books", "0002_uniquebook_detection_detectedbook"),
    ]

    operations = [
        migrations.AddField(
            model_name="uniquebook",
            name="detection",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="unique_books",
                to="books.detection",
            ),
        ),
    ]
