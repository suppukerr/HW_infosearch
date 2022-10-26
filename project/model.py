from transformers import AutoTokenizer, AutoModel
import torch
import os
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
# import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def indexing_documents(corpus):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    encoded_input = tokenizer(list(corpus), padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return corpus_embeddings

def vec_query(q, tokenizer=tokenizer):
    encoded_input = tokenizer([q], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def similarity(corpus, matrix, query):
    corpus['res_bert'] = cosine_similarity(matrix, query)
    A = np.array(corpus['res_bert'])
    ind = np.argsort(A, axis=0)
    B = np.array(corpus['answer'])
    return np.take_along_axis(B, ind, axis=0).tolist()