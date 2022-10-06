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

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[А-ЯЁа-яё]+')
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())  # и токенизируем
    lem = []  # для каждого документа создаём список и пополняем его леммами
    for word in tokens:
        lem.append(morph.parse(word)[0].normal_form)
    return ' '.join(lem)

def create(corpus):
    tips_data = pd.DataFrame(columns=['question', 'answer'])
    questions, answers = [], []

    for num, val in tqdm(enumerate(corpus)):
        ids = {item['text']: item['author_rating']['value'] for item in json.loads(corpus[num])['answers']}
        try:
            answer = max(ids, key=ids.get)  # самый популярный ответ
            questions.append(preprocess(json.loads(corpus[num])['question']))
            answers.append(answer)
        except:
            continue

    tips_data['question'], tips_data['answer'] = questions, answers
    return tips_data

def tf_idf_vect(corpus, query):
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(corpus).toarray()
    tf = count
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
    idf = tfidf_vectorizer.idf_
    query = preprocess(query)
    vect = tfidf_vectorizer.transform([query])
    return tf, idf, vect.toarray()

def bm_components(cur_tf, cur_idf, k, b, len_d, avg_len):
    b = cur_tf + (k*(1 - b + b*(len_d/avg_len)))
    a = cur_idf * cur_tf * (k + 1)
    return a/b

def bm_matrix(tf, idf):
    k = 2
    b = 0.75
    rows, cols, values = [], [], []
    len_d = [np.sum(i) for i in tf]  # ненулевых элементов в строке
    avg_len = np.mean(len_d)
    for i, j in tqdm(zip(*tf.nonzero())):
        cur_tf = tf[i, j]
        cur_idf = idf[j]
        cur_len_d = len_d[i]
        bm = bm_components(cur_tf, cur_idf, k, b, cur_len_d, avg_len)
        values.append(bm)
    matrix = sparse.csr_matrix((values, (tf.nonzero()[0], tf.nonzero()[1])))
    return matrix

def measure(lem_corpus, matrix, v_query):
    lem_corpus['res'] = pairwise_distances(matrix, v_query, metric='cosine')
    A = np.array(lem_corpus['res'])
    ind = np.argsort(A, axis=0)
    B = np.array(lem_corpus['answer'])
    return np.take_along_axis(B, ind, axis=0).tolist()