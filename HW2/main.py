from preprocess import prepare
import pandas as pd
import os
import argparse
from tfidf import *

def main(dir_data, query):
    if not os.path.isfile('friends_lemmas.csv'):
        lem_corpus = prepare(dir_data) # надо ещё базу по-другому прочитать чтобы сохранять название серии
        lem_corpus.to_csv('friends_lemmas.csv')
    else:
        lem_corpus = pd.read_csv('friends_lemmas.csv')
    matr_tfidf = vectorization(lem_corpus['text'])
    v_query = query_vector(query)
    distance = dist(lem_corpus, matr_tfidf, v_query)
    print('Серии, отсортированные по релевантности запросу:\n', '\n'.join(distance[:10])) #чтобы не отображать все, для примера берутся первые 10

if __name__ == '__main__':
    query = str(input('Введите запрос: '))
    main('friends-data', query)
