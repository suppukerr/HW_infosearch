from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

vectorizer = CountVectorizer(analyzer='word')
# import numpy as np
# import pandas as pd

# def vector(corpus):
#     vectorizer = CountVectorizer(analyzer='word')
#     X = vectorizer.fit_transform(corpus)
#     words = vectorizer.get_feature_names_out()
#     return words, X

def indexing_matrix(corpus, vectorizer=vectorizer):
    X = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names_out()
    return X.toarray(), words

def indexing_dict(matrix, words):
    inverted_dict = defaultdict(list)

    for i, word in enumerate(words):
        www = matrix[:, i]
        for j, count in enumerate(www):
            if www[j] != 0:
                inverted_dict[words[i]].append((j, www[j]))
    return inverted_dict

