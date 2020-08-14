dir_path = "/content/drive/My Drive/Microsoft_Hackathon/News Dataset/"
label_file = "labels_mod.csv"
fact_path = "sample-1M.jsonl"

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Importing packages
import re
import os
import sys
import numpy as np
import nltk
import random
import pickle
import csv

from google.colab import auth
from oauth2client.client import GoogleCredentials

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

MAX_ARTICLES = 2

article_list = []

with open(dir_path + label_file) as articles:
    article_list = articles.readlines()

media_maps = {
    "Twitter" : [0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0]
}

media_bias = {"left_bias":0 , "left_center_bias":1 , "least_biased":2 , "right_center_bias":3 , "right_bias":4}

for article_row in article_list[1:]:
    article = article_row.split(',')
    article[12] = media_bias[article[12]]
    arr = [float(x) for x in article[1:]]
    arr[9] = arr[9]/100
    arr[11] = arr[11]/4
    arr[12] = (arr[12] - 1)/4
    media_maps[article[0]] = arr

media_maps



len(media_maps)

new_folders = [x for x in os.listdir(dir_path) if "New folder" in x]

article_data = []
article_label = []
article_used_list = []
article_total_list = []

for new_folder in new_folders[:1]:
    for data_dir in os.listdir(dir_path + new_folder):
        for media_dir in os.listdir(dir_path + new_folder + "/" + data_dir):
            if media_dir in media_maps:
                for article in os.listdir(dir_path + new_folder + "/" + data_dir + "/" + media_dir)[:MAX_ARTICLES]:
                    print(article)
                    article_total_list.append(article)          
                    with open(dir_path + new_folder + "/" + data_dir + "/" + media_dir + "/" + article, "r") as article_file:
                            article_used_list.append(article)
                            sample = article_file.read()
                            article_data.append(sample)
                            article_label.append(media_dir)

article_used_list.sort()
for art in article_used_list:
    print(art)

article_total_list.sort()
for art in article_total_list:
    print(art)

len(article_data)

article_data

article_label

x_label = []
y_label = []

max_tok = 0

count = 0
with open(dir_path + "tweets.csv", "r", encoding="latin-1") as in_file:
    lines = in_file.readlines()
    for line in lines[1:]:
        if ("https" not in line):
            line = line.replace("\x93", "'")
            line = line.replace("\x94", "'")
            line = line.replace("\x92", "'")
            line = line.replace("&amp;", "&")
            x_label.append(line)
            y_label.append([(media_maps["Twitter"])[11], (media_maps["Twitter"])[12]])  
            count += 1

count = 0
with open(dir_path + "ExtractedTweets.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:

        temp_label = [(media_maps["Twitter"])[11], (media_maps["Twitter"])[12]]
        if (row[0] == "Democrat"):
            temp_label[0] = 0.0
        temp_label[1] = 0.25

        temp_sentence = row[2]
        if "RT" not in temp_sentence and "https" not in temp_sentence:
            temp_sentence = temp_sentence.replace("&amp;", "&")
            x_label.append(temp_sentence)
            y_label.append(temp_label.copy())
            print(temp_sentence)
            print(temp_label)

len(x_label)

x_label

"""Iterate through every sentence, add them to x_labels, and label each sentence."""

for i in range(len(article_data)):
    sent_size = len(word_tokenize(article_data[i]))

    if "Forbidden" not in article_data[i] and sent_size < 3200:

        x_label.append(article_data[i])
        y_label.append([media_maps[article_label[i]][11], media_maps[article_label[i]][12]])  
        max_tok = max(sent_size, max_tok)

print(i)

x_label

y_label

max_tok

vocabulary_size = 3600
embedding_size = 300
input_length = 3200

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(x_label)

len(list(tokenizer.word_index.keys()))

sequences = tokenizer.texts_to_sequences(x_label)
train_seqs = pad_sequences(sequences, maxlen=input_length, padding="post")

train_seqs

y_prime_1 = []
y_prime_2 = []

for label in y_label:
    y_prime_1.append(label[0])
    y_prime_2.append(label[1])

y_label = y_prime_1
print(y_label)

"""Bias Rating"""

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=input_length))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(128)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

history = model.fit(train_seqs, np.asarray(y_prime_1), epochs=10, batch_size=1, validation_split=0.1)

model.save(dir_path + "m_bias_e5_sig_specific_all.h5")

# saving
with open(dir_path + 'tokenizer_e5_sig_specific_all.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

right_input_text = "Mail-In Ballot fraud found in many elections. People are just now seeing how bad, dishonest and slow it is. Election results could be delayed for months. No more big election night answers? 1% not even counted in 2016. Ridiculous! Just a formula for RIGGING an Election...."
left_input_text = "We need to “transition to low carbon economy”"

test_sequences = tokenizer.texts_to_sequences([right_input_text])
test_sequences = pad_sequences(test_sequences, maxlen=3200, padding="post")
print(test_sequences)

prediction = model.predict(test_sequences)
print(prediction)

test_sequences = tokenizer.texts_to_sequences([left_input_text])
test_sequences = pad_sequences(test_sequences, maxlen=3200, padding="post")
prediction = model.predict(test_sequences)
print(prediction)

"""Factual Rating"""

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=input_length))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(128)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

print(y_prime_2)

model.fit(train_seqs, np.asarray(y_prime_2), epochs=10, batch_size=1)

model.save(dir_path + "m_fact_e5_sig_specific_all.h5")

right_input_text = "Mail-In Ballot fraud found in many elections. People are just now seeing how bad, dishonest and slow it is. Election results could be delayed for months. No more big election night answers? 1% not even counted in 2016. Ridiculous! Just a formula for RIGGING an Election...."
left_input_text = "Early Tuesday morning, our so-called president took to Twitter with one of his psycho rants about the Mueller investigation, calling Mueller conflicted and a damage to our criminal justice system. That tweet above marks around the 10th time since the midterm elections that Trump has used his favorite platform to mock Mueller and criticize the investigation that he loves to call a witch hunt. Does Trump really think that this crazy way of ranting and raving to his base via Twitter makes him look innocent? He must, because he was at it again this evening when he tweeted: The Mueller Witch Hunt is a total disgrace. They are looking at supposedly stolen Crooked Hillary Clinton Emails (even though they dont want to look at the DNC Server), but have no interest in the Emails that Hillary DELETED & acid washed AFTER getting a Congressional Subpoena! The Mueller Witch Hunt is a total disgrace. They are looking at supposedly stolen Crooked Hillary Clinton Emails (even though they dont want to look at the DNC Server), but have no interest in the Emails that Hillary DELETED & acid washed AFTER getting a Congressional Subpoena! I cant even count how many times hes mentioned Crooked Hillary this month. Its way past pathetic at this point, and his lunacy did not go unnoticed. Heres what the internet had to say: Trump is running scared and its glaringly obvious. Lets just all hope that Mueller hurries! Its time to lock Trump and his cronies up once and for all."
center_input_text = "Wendy Burke has had enough. Campaign advertisements bombard her favorite TV shows. Dozens of election pamphlets fill her mailbox. Every day, she gets several political calls on her cell phone and more on her landline. Strangers knock at her door seeking her vote. Its ridiculous, Burke, 47, said outside a shopping center in Palmdale, California. Ive had to block my calls. Welcome to the most expensive race in the hard-fought battle between Republicans and Democrats for control of the U.S. House of Representatives, which will be decided in Tuesdays elections. The blizzard of spending in Californias 25th district, a region stretching north and east of Los Angeles into the high desert of the Antelope Valley, stands out even during the most expensive congressional elections in U.S. history. Most of the money is funneled into non-stop advertising - on TV, radio, social media, yard signs, automated robocalls to cell phones and land lines, bumper stickers and a deluge of pamphlets stuffed into mailboxes. The mailers go in the trash, she said. I cant wait until this whole thing is over. The contest, a top Democratic target, has drawn more than $26 million in spending by candidates and outside groups since January 2017, according to a Reuters analysis of Federal Election Commission (FEC) data. It leads the 10 priciest House races, where a total of $238 million has been spent. tmsnrt.rs/2qrstCG"

test_sequences = tokenizer.texts_to_sequences([right_input_text])
test_sequences = pad_sequences(test_sequences, maxlen=3200, padding="post")
print(test_sequences)

prediction = model.predict(test_sequences)
print(prediction)

test_sequences = tokenizer.texts_to_sequences([left_input_text])
test_sequences = pad_sequences(test_sequences, maxlen=3200, padding="post")
prediction = model.predict(test_sequences)
print(prediction)

test_sequences = tokenizer.texts_to_sequences([center_input_text])
test_sequences = pad_sequences(test_sequences, maxlen=3200, padding="post")
prediction = model.predict(test_sequences)
print(prediction)