from rank_bm25 import BM25Okapi
from utils_func import corpus_processing, clustering, matrix_creation
from tqdm import tqdm
import numpy as np
import pandas as pd

class Retriever:
  def __init__(self, corpus:dict[str, str], clusters_dict:dict[str,str] = {}, k1:float=0.9, b:float=0.4):
    cleaned_corpus = corpus
    self.tokenized_corpus = [cleaned_corpus[key].split(" ") for key in corpus.keys()]
    self.bm25_model = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)    
    self.keys = list(corpus.keys())
    self.clusters_dict = clusters_dict


  def search(self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, score_function,**kwargs) -> dict[str, dict[str, float]]:
    results = {}
    for query_id, query in tqdm(queries.items(), desc="tests in progress"):
        # Process the query
        #cleaned_query = preprocess_corpus([query])
        cleaned_query = corpus_processing.clean_tokens(corpus_processing.nlp(query.lower()))
        cleaned_query = clustering.rewrite_text(cleaned_query, self.clusters_dict)
        tokenized_query = cleaned_query.split(" ")
        # Apply BM25 to get scores
        scores = self.bm25_model.get_scores(tokenized_query)
        # Sort the scores in descending order and save the results
        ordered_keys_index = np.argsort(scores)[::-1][:top_k]
        sorted_scores = {self.keys[i] : scores[i] for i in ordered_keys_index}
        results[query_id] = sorted_scores
    return results

class FullRetriever:
  def __init__(self, embeddings:pd.DataFrame, n_neighbors = 20, alpha:float=0.5, thresh = 0.8, metric = 'cosine', k1:float = 0.9, b:float = 0.4,coexistence_matrix:pd.DataFrame = None, thresh_prob = 0, compact_matrix = False):
    self.n_neighbors = n_neighbors
    self.alpha = alpha
    self.thresh = thresh
    self.metric = metric
    self.embeddings = embeddings
    self.coexistence_matrix = coexistence_matrix
    self.k1 = k1
    self.b = b
    self.cleaned_corpus = None
    self.thresh_prob = thresh_prob
    self.compact_matrix = compact_matrix


  def fit(self, corpus:dict[str, str], is_clean = False):
    if not is_clean:
      self.cleaned_corpus = corpus_processing.preprocess_corpus_dict(corpus)
    else:
      self.cleaned_corpus = corpus
    if self.coexistence_matrix is None:
      if self.compact_matrix or self.thresh_prob > 0:
        self.coexistence_matrix = matrix_creation.words_coexistence_probability_compact_parallel(self.cleaned_corpus, self.thresh_prob)
      else: 
        self.coexistence_matrix = matrix_creation.words_coexistence_probability(self.cleaned_corpus)

    words_in_common = list(set(self.coexistence_matrix.columns).intersection(set(self.embeddings.index)))
    self.embeddings = self.embeddings.loc[words_in_common]
    self.sim_mat = matrix_creation.get_similarity_matrix(self.embeddings, metric=self.metric, n_neighbors=self.n_neighbors)

    replaceable_words = clustering.get_replaceable_words(self.sim_mat, self.coexistence_matrix, alpha=self.alpha, thresh=self.thresh)

    word_graph = clustering.Graph(replaceable_words)
    self.clusters = word_graph.find_all_cycles()
    self.clust_dict = clustering.clusters_dict(self.clusters)

    self.rewritten_corpus = clustering.rewrite_corpus(self.cleaned_corpus, self.clust_dict)
    self.retriever = Retriever(self.rewritten_corpus, self.clust_dict, k1=self.k1, b=self.b)
    self.tokenized_corpus = self.retriever.tokenized_corpus

  def search(self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int, score_function,**kwargs) -> dict[str, dict[str, float]]:
    return self.retriever.search(corpus, queries, top_k, score_function, **kwargs)









      

