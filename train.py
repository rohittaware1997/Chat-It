import pandas as pd
import os
import json
import re
import numpy as np
import random
import string
import pickle
import nltk
import io
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag
import gensim
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity


import warnings
warnings.simplefilter('ignore')

def pre_process(questions):
    stop_words = stopwords.words("english")
    
    # Remove non english words
    questions = [re.sub('[^a-z(c++)(c#)]', ' ', x.lower()) for x in questions]
    # Tokenlization
    questions_tokens = [nltk.word_tokenize(t) for t in questions]
    # Removing Stop Words
    questions_stop = [[t for t in tokens if (t not in stop_words) and (3 < len(t.strip()) < 15)]
                      for tokens in questions_tokens]
    
    questions_stop = pd.Series(questions_stop)
    return questions_stop


def train_model(train_data):
    """Function trains and creates Word2vec Model using parsed
    data and returns trained model"""
    model = gensim.models.Word2Vec(train_data, min_count=2)
    return model

def train():
	path = 'Data/'
	data = pd.read_csv('dataset/StackOverflow.csv')

	nltk.download('stopwords')
	nltk.download('punkt')





	# Initial preprocessing training data
	questions = data['Question']
	questions_pp = pre_process(questions)


	data_tokens = pd.DataFrame({'Question': list(data['Question']),
	                            'Question_Tokens': questions_pp,
	                            'Answer': list(data['Answer']),
	                            'Class': list(data['Class'])
	                           })
	dict_language = {'0': 'python', '1': 'c++', '2': 'c#', '3': 'java', '4': 'ios', '5': 'android', '6': 'html', 
                 '7': 'jquery', '8': 'php', '9': 'javascript'}

	data_tokens['Question_Vectors'] = None
	data_tokens['Average_Pooling'] = None
    
	for key, value in dict_language.items():
		questions_data = list(data_tokens[data_tokens['Class'] == value]['Question_Tokens'])
		# Train model
		model_name = 'word2vec_model_' + value
		trained_model = train_model(questions_data)
		trained_model.save(model_name)
		print('Saved %s model successfully' % model_name)
    
		# Save Word2Vec model
		word2vec_pickle_path = path + 'stackoverflow_word2vec_' + value + '.bin'
		f = open(word2vec_pickle_path, 'wb')
		pickle.dump(trained_model, f) 
		f.close()
    
		model = gensim.models.KeyedVectors.load(word2vec_pickle_path)
    
		# Calculate the vectors for each question
		for i in range(len(data_tokens)):
			if data_tokens['Class'][i] == value:
				question_tokens = data_tokens['Question_Tokens'][i]
				question_vectors = []
				for token in question_tokens:
					try:
						vector = model[token]
						question_vectors.append(vector)
					except:
						continue
				# Vectors for each tokens
				data_tokens['Question_Vectors'][i] = question_vectors
				# Average Pooling of all tokens
				data_tokens['Average_Pooling'][i] = list(pd.DataFrame(question_vectors).mean())

	data_tokens['Question_Tokens'] = [" ".join(l) for l in data_tokens['Question_Tokens']]
	length = data_tokens['Question_Tokens'].apply(len)
	data_tokens = data_tokens.assign(Question_Length=length)
	data_tokens.head()

	# Export as data as JSON
	data_json = json.loads(data_tokens.to_json(orient='records'))

	with open(path + 'StackOverflow_Word2Vec.json', 'w') as outfile:
		json.dump(data_json, outfile)




