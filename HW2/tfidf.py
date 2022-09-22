from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=str.split)
import numpy as np
from preprocess import preprocess
from scipy.spatial.distance import cdist

def vectorization(corpus):
    X = vectorizer.fit_transform(corpus)
    return X.toarray()

def query_vector(query):
    query = preprocess(query)
    vect = vectorizer.transform([query])
    return vect.toarray()

def dist(corpus, matr_tfidf, v_query):
    corpus['res'] = cdist(matr_tfidf, v_query, metric='cosine')
    A = np.array(corpus['res'])
    ind = np.argsort(A, axis=0)
    B = np.array(corpus['name'])
    return np.take_along_axis(B, ind, axis=0).tolist()


