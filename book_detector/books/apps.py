from django.apps import AppConfig


class BooksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'books'


# class BooksConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'books'

#     def ready(self):
#         import books.signals  # Підключення сигналу
