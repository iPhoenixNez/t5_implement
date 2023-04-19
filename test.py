import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, T5ForConditionalGeneration, Trainer, TrainingArguments
autocast_questions = json.load(open('./competition/autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_questions = json.load(open('./competition/autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_questions]
autocast_questions = [q for q in autocast_questions if q['status'] == 'Resolved']

filtered_train_data = [example for example in autocast_questions if example['id'] not in test_ids and example['answer'] is not None]

import re
CLEANR = re.compile('<.*?>') 


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-qasc")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-qasc")

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def get_response(question, context, max_length=64):
  input_text = 'question: %s  context: %s' % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return cleanhtml(tokenizer.decode(output[0]))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel
'''
tokenizer_acc = AutoTokenizer.from_pretrained("distilbert-base-cased")
model_acc = AutoModel.from_pretrained("distilbert-base-cased")

def distilbertVec(ranswer, pred):
    # Tokenize and encode the sentences
    real_answer_tokens = tokenizer_acc.encode(ranswer, return_tensors="pt")
    prediction_tokens = tokenizer_acc.encode(pred, return_tensors="pt")

    # Generate fixed-length vector representations of the sentences
    with torch.no_grad():
        real_answer_vec = model_acc(real_answer_tokens)[0][:,0,:]
        prediction_vec = model_acc(prediction_tokens)[0][:,0,:]

    # Compute the cosine similarity between the two sentence vectors
    similarity = cosine_similarity(real_answer_vec, prediction_vec)
    return similarity[0][0]   
'''
def countVec(ranswer, pred):
    # Create a CountVectorizer object to convert the sentences to vectors of word counts
    vectorizer = CountVectorizer().fit_transform([ranswer, pred])
    # Calculate the cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    # Print the cosine similarity
    return cosine_sim

def brier_score(probabilities, answer_probabilities):
    return ((probabilities - answer_probabilities) ** 2).sum() / 2

from numpy import dot
from numpy.linalg import norm

for question_idx, ds_item in enumerate(filtered_train_data):

    if ds_item['qtype'] == 'mc':
        answer =ds_item['answer']
        question = ds_item['question']
        choice = ds_item['choices']
        # Print the context
        #print(answer)
        opt = ord(answer) - ord('A')
        ranswer = choice[opt]
        #print(ranswer)
        background = background[:min(len(background), 64)]
        pred = get_response(question, background)
        #print(pred)
        acc = countVec(ranswer, pred)
        #print(f"{acc:.2f}")
        # print(f"{distilbertVec(ranswer, pred):.2f}")
    elif ds_item['qtype'] == 't/f':
        question = ds_item['question']
        answer =ds_item['answer']
        background = ds_item['background']
        # Print the context
        print(answer)
        background = background[:min(len(background), 128)]
        pred = get_response(question+" Just answer 'Yes' or 'No'.", background)
        print(pred)
        acc = countVec(answer, pred)
        print(f"{acc:.2f}")
        #print(f"{distilbertVec(answer, pred):.2f}")
    elif ds_item['qtype'] == 'num':
        question = ds_item['question']
        answer =ds_item['answer']
        background = ds_item['background']
        # Print the context
        print(question)
        print(answer)
        background = background[:min(len(background), 128)]
        pred = get_response(question, background)
        print(pred)
        vector1 = np.array([float(pred)])
        vector2 = np.array([float(answer)])
        cos_sim = dot(vector1, vector2)/(norm(vector1)*norm(vector1))
        print(cos_sim)