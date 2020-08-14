# Flask Packages
import flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

# Tensorflow Packages
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence

# PyTorch/Transformers Packages
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# NLTK Packages
import nltk
from nltk.tokenize import sent_tokenize

# Azure Text Analytics Packages
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# General Python Packages
import json 
import numpy as np
import pickle
import random

# Preload all models
nltk.download('punkt')

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

model_bias = tf.keras.models.load_model("bias_model.h5")        
model_facts = tf.keras.models.load_model("fact_model.h5")        

with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

# Setup Azure Text Analytics API
key = ""
endpoint = ""

ta_credential = AzureKeyCredential(key)
client = TextAnalyticsClient(
        endpoint=endpoint, credential=ta_credential)

# Setup Flask
app = flask.Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True

# Flask Function
@app.route('/', methods=['POST', 'GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def home():
    bias = ""
    entity_links = ""
    input_text = ""
    output =  ""    
    sentiment = ""
    sent_arr = []

    # Get input from the client
    input_text = request.get_json()["Input"]

    # Perform sentiment analysis
    sentiment, sent_arr = sentiment_analysis(input_text[:5000])

    # Perform Entity extraction
    entity_links = entity_linking(input_text[:5000])

    # Perform Text Summarization
    output = text_summary(input_text)

    # Perform bias/factual rating
    bias = bias_fact_rating(input_text)

    # Generate sentence-by-sentence sentiment analysis
    sentences = compile_sentences(sent_arr)

    # Produce return JSON
    response_dict = {
        "entity" : entity_links,
        "sentiment" : sentiment,
        "summary" : output,
        "bias" : bias,
        "sentences" : sentences
    }

    return jsonify(response_dict)

def compile_sentences(sent_arr):
    sent_str = ""

    for i in range(len(sent_arr)):
        sent_str += (sent_arr[i])

    return sent_str

# Predicts the bias and factual rating of input
def bias_fact_rating(input_text):
    ratings = ""
    bias = 0
    factual = 0
    prediction = []

    # Tokenize the input text
    sequences = tok.texts_to_sequences([input_text])

    # Pad input text
    train_seqs = pad_sequences(sequences, maxlen=3200, padding="post")

    # Predict bias rating on input text
    bias = model_bias.predict(train_seqs)[0][0]

    # Predict factual rating on input text
    factual = model_facts.predict(train_seqs)[0][0]

    # Generate the rating string
    ratings = "Bias: %s\nFactual: %s\n\n" % (calculate_bias(bias), calculate_factual(factual))

    return ratings

# Sets bias based on threshold
def calculate_bias(raw_value):

    # left
    if raw_value < 0.2:
        return "Left"

    # lean left
    elif raw_value >= 0.2 and raw_value < 0.4:
        return "Lean Left"

    # center
    elif raw_value >= 0.4 and raw_value < 0.6:
        return "Moderate/Unbiased"

    # lean right
    elif raw_value >= 0.6 and raw_value < 0.8:
        return "Lean Right"

    # right
    elif raw_value >= 0.8:
        return "Right"            

    # right
    else:
        print("Error")
        return "Error"

# Produces the string percentage of factual rating
def calculate_factual(raw_value):
    return "%f%%" % ((raw_value) * 100)

# Performs Transformers Text Summarization
def text_summary(input_text):

    # Preprocess text
    preprocess_text = input_text.strip().replace("\n","")
    t5_prepared_Text = "summarize: " + preprocess_text

    # Tokenize text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    # Generate summary 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=5,
                                        max_length=1000,
                                        early_stopping=True)

    # Decode prediction
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output

# Obtains sentiment analysis
def sentiment_analysis(input_text):
    sent_overall = ""
    sent_arr = []

    # Prepare input and make API call
    documents = [input_text]
    response = client.analyze_sentiment(documents = documents)[0]

    # Generate the average of the overall sentiment
    sent_overall += "Document Sentiment: {}\n".format(response.sentiment)
    sent_overall += "Positive={0:.2f} | Neutral={1:.2f} | Negative={2:.2f} \n".format(
        response.confidence_scores.positive,
        response.confidence_scores.neutral,
        response.confidence_scores.negative,
    )

    # Obtain sentence by sentence sentiment
    for idx, sentence in enumerate(response.sentences):
        sentence_str = ""
        sentence_str += "\nSentence: {}\n".format(sentence.text)
        sentence_str += "Sentence {} sentiment: {}\n".format(idx+1, sentence.sentiment)
        sentence_str += "Positive={0:.2f} | Neutral={1:.2f} | Negative={2:.2f}\n".format(
            sentence.confidence_scores.positive,
            sentence.confidence_scores.neutral,
            sentence.confidence_scores.negative,
        )

    sent_arr.append(sentence_str)

    print(sent_overall)

    return sent_overall, sent_arr

# Obtain entity links
def entity_linking(input_text):

    try:
        # Prepare input and make the API call
        entities = ""
        documents = [input_text]
        result = client.recognize_linked_entities(documents = documents)[0]

        # Enumerate through all the entites and their source
        print("Linked Entities:\n")
        for entity in result.entities:
            entities += "Name: %s\nSource: %s\n\n" % (entity.name, entity.url)
            print("\tName: ", entity.name, "\tId: ", entity.data_source_entity_id, "\tUrl: ", entity.url,
            "\n\tData Source: ", entity.data_source)
            
    except Exception as err:
        print("Encountered exception. {}".format(err))        

    return entities

if __name__ == '__main__':
    app.run(threaded=True, port=5000) 