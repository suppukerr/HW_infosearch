import os
import pymorphy2
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd

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

def prepare(dir_data):
    corpus = []
    names = []
    for root, dirs, files in os.walk(dir_data): #смотрим на все файлы в директории
        for name in tqdm(files):
            with open(os.path.join(root, name), 'r', encoding="utf-8") as f:
                text = f.read().lower()
                lem = preprocess(text)
                lem = [word for word in lem if not word in stop_words]
                names.append(name)
                corpus.append(lem)
    friends_data = pd.DataFrame(columns=['name', 'text'])
    friends_data['name'], friends_data['text'] = names, corpus
    return friends_data
