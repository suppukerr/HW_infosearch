import os
import pymorphy2
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import json
from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity
import argparse
from model import *
from creating import *
import pickle

import streamlit as st

dir_data = 'data.jsonl'

def main(query):
    if not os.path.isfile('love_questions.csv'):
        with open(dir_data, 'r', encoding='utf-8') as f:
            corpus = list(f)[:5000]
        lem_corpus = create(corpus)
        lem_corpus = lem_corpus.dropna(axis=0, subset=['question'])
        lem_corpus.to_csv('love_questions.csv')
    else:
        lem_corpus = pd.read_csv('love_questions.csv', index_col=0)
    lem_corpus = lem_corpus.dropna(axis=0, subset=['question'])
    if :
        if not os.path.isfile('matrix_bertina.obj'):
            matrix = indexing_documents(lem_corpus['non_prep_question'])
            filehandler = open('matrix_bertina.obj', 'wb')
            pickle.dump(matrix, filehandler)
        else:
            filehandler = open('matrix_bertina.obj', 'rb')
            matrix = pickle.load(filehandler)
        vec = vec_query(query)
        meas = similarity(lem_corpus, matrix, vec)
        print('Ответы, полученные бертом и отсортированные по релевантности запросу:\n',
               '\n'.join(meas[:10]))

if __name__ == '__main__':
    main(str(input('Введите запрос: ')))