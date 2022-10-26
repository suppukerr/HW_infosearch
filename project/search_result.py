import os
from tqdm import tqdm
import pandas as pd
import json
from scipy import sparse
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity
from model import *
from creating import *
import pickle

dir_data = 'data.jsonl'
lem_corpus = pd.read_csv('love_questions.csv', index_col=0)
lem_corpus = lem_corpus.dropna(axis=0, subset=['question'])
filehandler1 = open('matrix_bertina.obj', 'rb')
matrix_bert = pickle.load(filehandler1)
filehandler2 = open('matrix_bm.obj', 'rb')
matrix_bm = pickle.load(filehandler2)
def search(query, metrics):
    start = time.time()
    if metrics == 'bert':
        vec = vec_query(query)
        meas = similarity(lem_corpus, matrix_bert, vec)
        result = meas[:10]
    else:
        tf, idf, v_query = tf_idf_vect(lem_corpus['question'], query)
        closeness = measure(lem_corpus, matrix_bm, v_query)
        result = closeness[:10]
    end = time.time()
    return result, end-start

if __name__ == '__main__':
    main(str(input('Введите запрос: ')))
