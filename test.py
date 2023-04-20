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
import decimal
for question_idx, ds_item in enumerate(filtered_train_data):
    predictions = []
    answers = []
    qtypes = []
    if ds_item['qtype'] == 'mc':
        answer =ds_item['answer']
        question = ds_item['question']
        choice = ds_item['choices']  
        background = ds_item['background']
        background = background[:min(len(background), 64)]
        # Print the context
        #print(answer)
        print(question)
        print(background)
        print(answer)
        pred = get_response(question, background)
        opt = ord(answer) - ord('A')
        ranswer = choice[opt]
        print(ranswer)
        prediction_acc = []
        print("pred"+pred)
        for i in range(len(choice)):
            prediction_acc.append(countVec(choice[i], pred))# get a list of the accuray for every choice
            print("choice "+choice[i])
        # Convert list to decimals
        print(prediction_acc)
        prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
        summ = sum(prediction_acc)
        if summ == 0:
            summ = 1e-5
        # Divide each decimal by the sum of decimals
        prediction_acc = [x / summ for x in prediction_acc]
        prediction_arr = np.array(prediction_acc)
        answer_arr = np.zeros(len(choice))
        answer_arr[opt] = 1
        print(prediction_arr)
        print(answer_arr)
        predictions.append(prediction_arr)
        answers.append(answer_arr)
        qtypes.append(ds_item['qtype'])
    elif ds_item['qtype'] == 't/f':#G 28, G30
        question = ds_item['question']
        answer =ds_item['answer']
        background = ds_item['background']
        # Print the context
        print(question)
        print(background)
        print(answer)
        background = background[:min(len(background), 128)]
        pred = get_response(question+" Just answer 'Yes' or 'No'.", background)
        print(pred)
        #acc = countVec(answer, pred)
        prediction_acc = []
        prediction_acc.append(countVec("yes", pred))# get a list of the accuray for every choice
        prediction_acc.append(countVec("no", pred))# get a list of the accuray for every choice
        # Convert list to decimals
        print(prediction_acc)
        prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
        summ = sum(prediction_acc)
        if summ == 0:
            summ = 1e-5
        # Divide each decimal by the sum of decimals
        prediction_acc = [x / summ for x in prediction_acc]
        prediction_arr = np.array(prediction_acc)
        #print(f"{acc:.2f}")
        print(prediction_arr)
        if answer == 'yes':
            answer_acc = [1, 0]
            answer_acc = np.array(answer_acc)
        else:
            answer_acc = [0, 1]
            answer_acc = np.array(answer_acc)
        predictions.append(prediction_arr)
        answers.append(answer_arr)
        qtypes.append(ds_item['qtype'])
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
        predictions.append(cos_sim)
        answers.append(answer)
        qtypes.append(ds_item['qtype'])

'''

import decimal
def cal_posb():
    predictions = []
    answers = [] 
    for question_idx, ds_item in enumerate(filtered_train_data):
        if ds_item['qtype'] == 'mc':
            answer =ds_item['answer']
            question = ds_item['question']
            choice = ds_item['choices']  
            background = ds_item['background']
            background = background[:min(len(background), 128)]
            # Print the context
            #print(answer)
            pred = get_response(question, background)
            choices = choice.values.tolist()
            opt = ord(answer) - ord('A')
            prediction_acc = []
            for i in range(len(choices)):
                prediction_acc[i] = countVec(choices[i], pred)# get a list of the accuray for every choice
            prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
            summ = sum(prediction_acc)
            # Divide each decimal by the sum of decimals
            prediction_acc = [x / summ for x in prediction_acc]
            prediction_arr = np.array(prediction_acc)
            answer_arr = np.zeros(len(choices))
            answer_arr[opt] = 1
        predictions.append(prediction_arr)
        answers.append(answer_arr)

import numpy as np
predicted_probs = np.array(a)
true_probs = np.array([0, 0, 1, 0])
score = brier_score(predicted_probs, true_probs)#the list a, [1, 0, 0, 0] for example
print(score)


import string 
def calculate_answer_probability(answer_arr, choice_arr, qtype):
    """
    Give the answer array, calculate the possibilities of each answer given the choices and answer_arr
    """
    # Process the answer
    if qtype == "t/f":
        # unique: 2x1 array
        # counts: 2x1 array
        choice_arr_tf = ["yes", "no"]
        N = len(answer_arr)
        unique, counts = np.unique(answer_arr, return_counts=True)
        choice_len = len(choice_arr_tf)
        new_counts = np.zeros(choice_len)
        for i in range(choice_len):
            curr_choice = choice_arr_tf[i]
            if curr_choice not in unique:
                new_counts[i] = 0
            else:
                new_counts[i] = counts[curr_choice == unique]

        # Probability of yes/no
        return new_counts / N
    
    elif qtype == "mc":
        # Total number of prediciton
        # Should be 10
        # choice_arr = process_choice(choice_arr)
        N = len(answer_arr)
        unique, counts = np.unique(answer_arr, return_counts=True)
        choice_len = len(choice_arr)
        # Generate answer choice
        answer_choice = list(string.ascii_uppercase)[:choice_len]
        # Create a new counts array to hold the prob of all answers
        new_counts = np.zeros(len(choice_arr))
        for i in range(choice_len): 
            curr_choice = answer_choice[i]
            if curr_choice not in unique: 
                new_counts[i] = 0
            else:
                new_counts[i] = counts[curr_choice == unique]
 
        # Probability of each answer in an array
        return new_counts / N

    elif qtype == "num":
        # Number questions, return the number only  
        return answer_arr[0]
        '''