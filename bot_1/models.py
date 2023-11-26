from django.db import models

class ChatMessage(models.Model):
    message = models.TextField()