from rank_bm25 import BM25Okapi
from utils_func import corpus_processing, clustering, matrix_creation
from tqdm import tqdm
import numpy as np
import pandas as pd
import fasttext
from sklearn.neighbors import NearestNeighbors
from typing import Literal


class Retriever:
  def __init__(self, corpus:dict[str, str], fasttext_model, clusters_dict:dict[str,str] = {}, thresh:float = 0.75,k1:float=0.9, b:float=0.4, embeddings:pd.DataFrame = None):
    cleaned_corpus = corpus
    self.tokenized_corpus = [cleaned_corpus[key].split() for key in corpus.keys()]
    self.bm25_model = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)    
    self.keys = list(corpus.keys())
    self.clusters_dict = clusters_dict
    self.fasttext_model = fasttext_model
    self.thresh = thresh
    self.embeddings = embeddings
    

  def search(self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, score_function,**kwargs) -> dict[str, dict[str, float]]:
    results = {}
    for query_id, query in tqdm(queries.items(), desc="tests in progress"):
        # Process the query
        #cleaned_query = preprocess_corpus([query])
        cleaned_query = corpus_processing.clean_tokens(corpus_processing.nlp(query.lower()))
        cleaned_query = clustering.rewrite_text(cleaned_query, self.clusters_dict, self.fasttext_model, thresh = self.thresh)
        tokenized_query = cleaned_query.split()

        # Apply BM25 to get scores
        scores = self.bm25_model.get_scores(tokenized_query)

        # Sort the scores in descending order and save the results
        ordered_keys_index = np.argsort(scores)[::-1][:top_k]
        sorted_scores = {self.keys[i] : scores[i] for i in ordered_keys_index}
        results[query_id] = sorted_scores
    return results

class UCFIRe:
  def __init__(self, embeddings:pd.DataFrame, fasttext_model, n_neighbors = 20, alpha:float=0.5, thresh = 0.8, metric:Literal['euclidean', 'cosine'] ='cosine', k1:float = 0.9, b:float = 0.4, thresh_prob:float = 0.0):
    self.n_neighbors = n_neighbors
    self.alpha = alpha
    self.thresh = thresh
    self.embeddings = embeddings
    self.k1 = k1
    self.b = b
    self.cleaned_corpus = None
    self.thresh_prob = thresh_prob
    self.fasttext_model = fasttext_model
    self.metric = metric

    self.tokenized_corpus = None
    self.retriever = None


  def fit(self, corpus, is_clean = False, knn_method:Literal['exact', 'faiss'] = 'exact'):
    if not is_clean:
      self.cleaned_corpus = corpus_processing.preprocess_corpus_dict(corpus)
    else:
      self.cleaned_corpus = corpus

    replaceable_words = clustering.get_replaceable_words(self.cleaned_corpus, self.embeddings, self.thresh_prob, self.metric, self.n_neighbors, self.alpha, self.thresh, knn_method)

    word_graph = clustering.Graph(replaceable_words)
    print('finding graph components...')
    self.clusters = word_graph.find_all_cycles()
    print('grap components found')

    self.clust_dict = clustering.clusters_dict(self.clusters)

    self.rewritten_corpus = clustering.rewrite_corpus(self.cleaned_corpus, self.clust_dict)
    self.retriever = Retriever(self.rewritten_corpus, self.fasttext_model,self.clust_dict, k1=self.k1, b=self.b, embeddings = self.embeddings)
    self.tokenized_corpus = self.retriever.tokenized_corpus

  def switch_fasttext_model(self, fasttext_model):
    self.fasttext_model = fasttext_model
    self.retriever.fasttext_model = fasttext_model

  def search(self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, score_function,**kwargs) -> dict[str, dict[str, float]]:
    return self.retriever.search(corpus, queries, top_k, score_function, **kwargs)
