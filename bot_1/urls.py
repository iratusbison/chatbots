from django.urls import path
from . import views

urlpatterns = [
    path('bot', views.chatbot, name='chatbot'),
    path('', views.delete_chat, name='delete_chat'),
]
