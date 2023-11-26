from django.urls import path
from . import views

urlpatterns = [
    # Other URL patterns...
    path('', views.process_message, name='process_message'),
     path('clear/', views.clear_chat_history, name='clear_chat_history'),
]
