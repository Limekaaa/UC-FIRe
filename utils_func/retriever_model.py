from rank_bm25 import BM25Okapi
import corpus_processing
from tqdm import tqdm
import numpy as np
import clustering

class Retriever:
  def __init__(self, corpus:dict[str, str], clusters_dict:dict[str,str], k1:float=0.9, b:float=0.4):
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
        cleaned_query = corpus_processing.clean_tokens(corpus_processing.nlp(query))
        cleaned_query = clustering.rewrite_text(cleaned_query, self.clusters_dict)
        tokenized_query = cleaned_query.split(" ")
        # Apply BM25 to get scores
        scores = self.bm25_model.get_scores(tokenized_query)
        # Sort the scores in descending order and save the results
        ordered_keys_index = np.argsort(scores)[::-1][:top_k]
        sorted_scores = {self.keys[i] : scores[i] for i in ordered_keys_index}
        results[query_id] = sorted_scores
    return results