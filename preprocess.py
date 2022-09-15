import os
import pymorphy2
from nltk.corpus import stopwords
from tqdm import tqdm
# import pandas as pd
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[А-ЯЁа-яё]+')
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def prepare(dir_data):
    corpus = []
    for root, dirs, files in os.walk(dir_data): #смотрим на все файлы в директории
        for name in tqdm(files):
            with open(os.path.join(root, name), 'r', encoding="utf-8") as f:
                text = f.read().lower()
                tokens = tokenizer.tokenize(text) # и токенизируем
                lem = [] # для каждого документа создаём список и пополняем его леммами
                for word in tokens:
                    lem.append(morph.parse(word)[0].normal_form)
                corpus.append(' '.join(lem))
    corpus = [word for word in corpus if not word in stop_words]
    return corpus

# dir_data = 'friends-data'
# lem_corpus = prepare(dir_data)
# print(len(lem_corpus))
