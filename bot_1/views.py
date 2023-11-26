import json
import nltk
import random
from django.shortcuts import render
from .models import ChatMessage 


# Load intents from the intents.json file
with open('bot_1/intents.json') as file:
    intents = json.load(file)


 

chat_history = []

def process_message(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')  # Assuming the input field is named 'message'

        # Add user message to chat history
        chat_history.append({'message': user_message, 'sender': 'user'})

        # Find the appropriate response based on user input
        bot_response = get_response_from_intents(user_message, intents)

        # Add bot response to chat history
        chat_history.append({'message': bot_response, 'sender': 'bot'})

        return render(request, 'chatbot.html', {'chat_history': chat_history})

    return render(request, 'chatbot.html', {'chat_history': chat_history})

'''
def process_message(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')  # Assuming the input field is named 'message'
        
        # Find the appropriate response based on user input and intents
        response = get_response_from_intents(user_message, intents)
        
        return render(request, 'chatbot.html', {'response': response})

    return render(request, 'chatbot.html', {'response': None})
'''
def get_response_from_intents(user_message, intents):
    user_tokens = nltk.word_tokenize(user_message.lower())  # Tokenize user input
    matching_intents = []

    # Compare user tokens with intent patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            intent_tokens = nltk.word_tokenize(pattern.lower())
            if all(token in intent_tokens for token in user_tokens):
                matching_intents.append(intent)

    # Choose a random response from matching intents (if any)
    if matching_intents:
        selected_intent = random.choice(matching_intents)
        response = random.choice(selected_intent['responses'])
        return response

    return "Apologies, I didn't understand that."

def clear_chat_history(request):
    global chat_history
    chat_history = []  # Clear the chat history (reset to an empty list)
    return render(request, 'chatbot.html', {'chat_history': chat_history})