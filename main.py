from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import random
import nltk
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Load dataset and model
with open('intents.json') as file:
    data = json.load(file)

# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()  # Menggunakan split untuk tokenisasi
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Extract patterns and tags
all_words = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = clean_text(pattern)
        all_words.extend(tokens)
    tags.append(intent['tag'])

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Predict the class of the user input
def predict_class(sentence):
    tokens = clean_text(sentence)
    bag = [1 if word in tokens else 0 for word in all_words]
    bag = np.array(bag).reshape(1, -1)
    prediction = model.predict(bag, verbose=0)
    tag_index = np.argmax(prediction)
    tag = tags[tag_index]
    return tag

# Get response based on the predicted tag
def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# FastAPI app setup
app = FastAPI()


# Mengizinkan semua origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origins
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua metode (GET, POST, PUT, DELETE, dll.)
    allow_headers=["*"],  # Mengizinkan semua header
)


# Define request model
class UserInput(BaseModel):
    message: str

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API"}

# Define the predict endpoint
@app.post("/predict/")
def predict(input: UserInput):
    tag = predict_class(input.message)
    response = get_response(tag)
    return {"response": response}
