from django.urls import path
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LogoutView

from . import views

urlpatterns = [
    path('', views.detect_books, name='detect_books'),
    path('save-books/', views.save_books, name='save_books'),
    path('unique-books/', views.unique_books_list, name='unique_books_list'),
    path('unique-books/<int:pk>/', views.unique_book_detail, name='unique_book_detail'),

    path('detection/<int:pk>/', views.detection_detail, name='detection_detail'),
    path('detection/<int:detection_id>/save-books/', views.save_unique_books_from_detection, name='save_unique_books_from_detection'),

    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),

    path('user-books/', views.user_unique_books, name='user_unique_books'),

    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='books/login.html'), name='login'),
    # path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
