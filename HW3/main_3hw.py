from creating import *
import pandas as pd
import os
import pandas as pd
import pickle

k = 2
b = 0.75

def main(query, dir_data='data.jsonl'): # понизить регистр у запроса
    with open(dir_data, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    if not os.path.isfile('love_questions.csv'):
        lem_corpus = create(corpus)  # надо ещё базу по-другому прочитать чтобы сохранять название серии
        lem_corpus.to_csv('love_questions.csv')
    else:
        lem_corpus = pd.read_csv('love_questions.csv', index_col=0)
    lem_corpus = lem_corpus.dropna(axis=0, subset=['question'])
    tf, idf, v_query = tf_idf_vect(lem_corpus['question'], query)
    if not os.path.isfile('matrix.obj'):
        matrix = bm_matrix(tf, idf)
        filehandler = open('matrix.obj', 'wb')
        pickle.dump(matrix, filehandler)
    else:
        filehandler = open('matrix.obj', 'rb')
        matrix = pickle.load(filehandler)
    closeness = measure(lem_corpus, matrix, v_query)
    print('Ответы, отсортированные по релевантности запросу:\n',
           '\n'.join(closeness[:10]))  # чтобы не отображать все, для примера берутся первые 10

if __name__ == '__main__':
    main(str(input('Введите запрос: ')))