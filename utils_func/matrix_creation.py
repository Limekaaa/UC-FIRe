import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import multiprocessing as mp
from multiprocessing import Pool

from typing import Dict, Set, Any, Literal

from scipy.sparse import csc_matrix, lil_matrix
import os
import faiss

def get_unique_words(corpus:dict[str, str]) -> set:
    """
    Function to find the unique words in a corpus
    :param corpus: dict[int, str] - a dictionary with the key being the document id and the value being the document text
    :return: set - a set of unique words in the corpus
    """
    unique_words = set()
    for doc_id in tqdm(corpus, desc="Getting unique words"):
        text = corpus[doc_id]
        text = text.split()
        for word in text:
            unique_words.add(word)
    return unique_words

def get_word_presence(corpus:dict[str, str]) -> dict[str, set[int]]:
    """
    Function to get the presence of each word in each document
    :param corpus: dict[int, str] - a dictionary with the key being the document id and the value being the document text
    :return: pd.DataFrame - a dataframe with the presence of each word in each document
    """
    unique_words = list(get_unique_words(corpus))
    word_presence = {unique_words[i]: set() for i in range(len(unique_words))}

    for doc_id in tqdm(corpus, desc="Getting word presence"):
        text = corpus[doc_id]
        text = text.split()
        for word in text:
            word_presence[word].add(doc_id)
    return word_presence

def words_coexistence_probability_compact(corpus:dict[str, str], words:list[str], thresh_prob = 0) -> csc_matrix:
    """
    Function to calculate the probability of coexistence of each pair of words in the corpus in a compact way
    :param corpus: dict[str, str] - a dictionary with the key being the document id and the value being the document text
    :param thresh_prob: float - the threshold probability to consider the coexistence of two words, allow to reduce the size of the matrix
    :return: pd.DataFrame - a dataframe with the probability of coexistence of each pair of words in the corpus
    """
    word_presence = get_word_presence(corpus)
    unique_words = words
    
    row = []
    col = []
    data = []

    for word2 in tqdm(range(len(unique_words)), desc= "Calculating coexistence probability"):
        #dico[unique_words[word2]] = dict()
        for word1 in range(word2, len(unique_words)):
            inter = len(word_presence[unique_words[word1]].intersection(word_presence[unique_words[word2]]))
            if inter > 0:
                prob = inter/max(len(word_presence[unique_words[word1]] | word_presence[unique_words[word2]]), 1)
                if prob > thresh_prob:
                    #dico[unique_words[word2]][unique_words[word1]] = prob
                    row.append(word2)
                    col.append(word1)
                    data.append(prob)

                    if word1 != word2:
                        row.append(word1)
                        col.append(word2)
                        data.append(prob)
                    
    return csc_matrix((data, (row, col)), shape=(len(unique_words), len(unique_words)))

def get_similarity_matrix(embeddings:pd.DataFrame, metric:Literal['euclidean', 'cosine'] = 'euclidean', n_neighbors:int = 5, method:Literal['exact', 'faiss'] = 'exact') -> csc_matrix:
    """
    Function to calculate the similarity matrix between all words
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :return: pd.DataFrame - a dataframe with the similarity score between all words
    """
    if method != 'exact' and metric == 'euclidean':
        raise ValueError('Euclidean distance is not supported by the chosen method, if you want to use euclidean distance, please choose the exact method')
    
    if method == 'exact':
        print('fitting Nearest Neighbors')
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs = -1).fit(embeddings)
        print('End of fitting Nearest Neighbors')
        print('getting distances')
        distances, indices = neighbors.kneighbors(embeddings)  
        print('end of getting distances')
    elif method == 'faiss' and metric == 'cosine':
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        # Normalize the embeddings to unit length (for cosine similarity)
        print("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)  # Normalize the embeddings in place
        print("Embeddings normalized.")
        # Create a Faiss index for cosine similarity (using inner product)
        print("Creating Faiss index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        print("Faiss index created.")

        # Add the normalized embeddings to the index
        index.add(embeddings)
        
        print('getting distances')
        # Perform the nearest neighbor search
        distances, indices = index.search(embeddings, n_neighbors)    
        print('end of getting distances')
    else:
        raise ValueError('The method or the metric chosen is not supported')


    filled_mat = lil_matrix((len(embeddings), len(embeddings)))

    if metric == 'euclidean':
        for i in range(len(embeddings)):
            max_dist = np.max(distances)
            filled_mat[i, indices[i]] = 1-(distances[i]/max_dist)
            filled_mat[indices[i], i] = 1-(distances[i]/max_dist)
    else:
        if method == 'exact':
            for i in tqdm(range(len(embeddings)), desc='filling similarity matrix'):
                filled_mat[i, indices[i]] = 1-distances[i]
                filled_mat[indices[i], i] = 1-distances[i]
        else:
            for i in tqdm(range(len(embeddings)), desc='filling similarity matrix'):
                filled_mat[i, indices[i]] = distances[i]
                filled_mat[indices[i], i] = distances[i]

    filled_mat = filled_mat.tocsc()

    return filled_mat

# coexistence probability calculation in parallel ______________________________________________________________________________________________________________________________

# Worker initializer to set globals in each process.
def init_worker(shared_word_presence, shared_unique_words, shared_thresh_prob):
    global word_presence, unique_words, thresh_prob
    word_presence = shared_word_presence
    unique_words = shared_unique_words
    thresh_prob = shared_thresh_prob

# Worker function: for a given word2 index, compute all (word2, word1) pairs.
def process_word2(word2):
    local_entries = []
    for word1 in range(word2, len(unique_words)):
        # Retrieve the sets of document IDs for the two words.
        set1 = word_presence[unique_words[word1]]
        set2 = word_presence[unique_words[word2]]
        inter = len(set1.intersection(set2))
        if inter > 0:
            union = len(set1.union(set2))
            prob = inter / max(union, 1)
            if prob > thresh_prob:
                # Append the (word2, word1) entry.
                local_entries.append((word2, word1, prob))
                # If the pair is not on the diagonal, also append the symmetric entry.
                if word1 != word2:
                    local_entries.append((word1, word2, prob))
    return local_entries

def words_coexistence_probability_compact_parallel(corpus: dict[str, str], words: list[str], thresh_prob=0.0) -> csc_matrix:
    """
    Calculate the probability of coexistence of each pair of words in parallel.

    :param corpus: dict[int, str] - a dictionary with document IDs as keys and document texts as values.
    :param words: list[str] - a list of words for which to compute the probabilities.
    :param thresh_prob: float - the threshold probability below which pairs are ignored.
    :return: csc_matrix - the sparse matrix of coexistence probabilities.
    """
    # Build a mapping from each word to the set of document IDs in which it appears.
    #word_presence = get_word_presence(corpus)
    word_presence = get_word_presence(corpus)
    unique_words = words  # assuming 'words' is the list of unique words

    # You can set num_processes to a specific number (e.g. os.cpu_count()) or leave it as None.
    num_processes = os.cpu_count()

    # Use a Pool with an initializer so that each worker gets the shared data.
    with Pool(processes=num_processes,
              initializer=init_worker,
              initargs=(word_presence, unique_words, thresh_prob)) as pool:
        # Distribute the iterations of the outer loop. Each task is a word2 index.
        total = len(unique_words)
        results = list(tqdm(pool.imap_unordered(process_word2, range(total)),
                            total=total,
                            desc="Calculating coexistence probability"))

    # Combine all the entries returned by the worker processes.
    row, col, data = [], [], []
    for local_entries in results:
        for r, c, d in local_entries:
            row.append(r)
            col.append(c)
            data.append(d)

    # Create and return the sparse matrix.
    return csc_matrix((data, (row, col)), shape=(len(unique_words), len(unique_words)))