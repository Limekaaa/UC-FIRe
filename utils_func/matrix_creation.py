import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import multiprocessing as mp
from typing import Dict, Set, Any

from scipy.sparse import csc_matrix, lil_matrix
import os

class coex_matrix:
    def __init__(self, dico:dict[str, dict[str, float]], unique_words:list[str]):
        self.dico = dico
        self.index = unique_words
        self.columns = unique_words
    def __getitem__(self, key:str):
        return self.dico[key]

    '''
    def loc(self, word):
        temp = np.zeros(len(self.index))
        for i in list(self.dico[word].keys()):
            temp[self.index.index(i)] = self.dico[word][i]
        return pd.Series(temp, index=self.index)
    '''

    def loc(self, word):
        index = list(self.dico[word].keys())
        vals = [self.dico[word][i] for i in index]

        return pd.Series(vals, index=index)

'''
row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
'''



def get_unique_words(corpus:dict[int, str]) -> set:
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

def get_word_presence(corpus:dict[int, str]) -> dict[str, set[int]]:
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

def words_coexistence_probability_compact(corpus:dict[int, str], words:list[str], thresh_prob = 0) -> csc_matrix:
    """
    Function to calculate the probability of coexistence of each pair of words in the corpus in a compact way
    :param corpus: dict[int, str] - a dictionary with the key being the document id and the value being the document text
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


def words_coexistence_probability(corpus:dict[int, str]) -> pd.DataFrame:
    """
    Function to calculate the probability of coexistence of each pair of words in the corpus
    :param corpus: dict[int, str] - a dictionary with the key being the document id and the value being the document text
    :return: pd.DataFrame - a dataframe with the probability of coexistence of each pair of words in the corpus
    """
    word_presence = get_word_presence(corpus)
    unique_words = list(get_unique_words(corpus))
    #word_coexistence = pd.DataFrame(0.0,index=unique_words, columns=unique_words)
    print("Calculating coexistence probability")
    word_coexistence = np.array([[len(word_presence[unique_words[word1]].intersection(word_presence[unique_words[word2]]))/max(len(word_presence[unique_words[word1]] | word_presence[unique_words[word2]]), 1) for word1 in range(len(unique_words))] for word2 in tqdm(range(len(unique_words)))])
    print("Creating dataframe")
    word_coexistence = pd.DataFrame(word_coexistence, index=unique_words, columns=unique_words)
    return word_coexistence



def process_word(args: tuple) -> tuple:
    """
    Worker function to compute coexistence probabilities for a single word.
    """
    word2, thresh_prob = args
    global word_presence, unique_words, word_counts
    entries = {}
    for word1 in unique_words:
        # Calculate intersection
        inter = len(word_presence[word1].intersection(word_presence[word2]))
        if inter > 0:
            # Calculate union using precomputed lengths
            len1 = word_counts[word1]
            len2 = word_counts[word2]
            union = len1 + len2 - inter
            prob = inter / max(union, 1)
            if prob > thresh_prob:
                entries[word1] = prob
    return (word2, entries)

def init_worker(shared_word_presence: Dict[str, Set[int]], 
                shared_unique_words: list, 
                shared_word_counts: Dict[str, int]):
    """
    Initializer function for worker processes to set up global variables.
    """
    global word_presence, unique_words, word_counts
    word_presence = shared_word_presence
    unique_words = shared_unique_words
    word_counts = shared_word_counts

def words_coexistence_probability_compact_parallel(corpus: Dict[int, str], 
                                                  thresh_prob: float = 0.0) -> coex_matrix:
    """
    Parallelized function to calculate the probability of coexistence of each pair of words.
    """
    # Precompute necessary data structures
    word_presence = get_word_presence(corpus)
    unique_words = list(get_unique_words(corpus))
    word_counts = {word: len(docs) for word, docs in word_presence.items()}
    
    # Prepare arguments for each word
    args_list = [(word, thresh_prob) for word in unique_words]
    
    # Create a pool of workers
    with mp.Pool(
        initializer=init_worker,
        initargs=(word_presence, unique_words, word_counts)
    ) as pool:
        # Process each word in parallel with tqdm progress bar
        results = list(tqdm(
            pool.imap(process_word, args_list),
            total=len(unique_words),
            desc="Processing words"
        ))
    
    # Merge results into the final dictionary
    dico = {word2: entries for word2, entries in results}
    return coex_matrix(dico, unique_words)

def cosine_similarity(word1:str, word2:str, embeddings:pd.DataFrame) -> float:
    """
    Function to calculate the similarity score between two words
    :param word1: str - the first word
    :param word2: str - the second word
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :return: float - the similarity score between the two words
    """

    return np.dot(embeddings.loc[word1], embeddings.loc[word2])/(np.linalg.norm(embeddings.loc[word1])*np.linalg.norm(embeddings.loc[word2]))

def distance(word1:str, word2:str, embeddings:pd.DataFrame) -> float:
    """
    Function to calculate the distance between two words
    :param word1: str - the first word
    :param word2: str - the second word
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :return: float - the distance between the two words
    """
    return np.linalg.norm(embeddings.loc[word1] - embeddings.loc[word2])


def get_nearest_neighbors(word:str, embeddings:pd.DataFrame, n_neighbors:int=5, metric = 'cosine') -> list:
    """
    Function to get the nearest neighbors of a word
    :param word: str - the word to get the nearest neighbors
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :param n_neighbors: int - the number of neighbors to get
    :return: list - a list with the nearest neighbors
    """
    print('fitting Nearest Neighbors')
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(embeddings)
    print('End of fitting Nearest Neighbors')
    return neighbors.kneighbors(embeddings.loc[word].values.reshape(1,-1))


def get_similarity_matrix(embeddings:pd.DataFrame, metric:str = 'euclidean', n_neighbors:int = 5) -> csc_matrix:
    """
    Function to calculate the similarity matrix between all words
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :return: pd.DataFrame - a dataframe with the similarity score between all words
    """
    print('fitting Nearest Neighbors')
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs = -1).fit(embeddings)
    print('End of fitting Nearest Neighbors')
    print('getting distances')
    distances, indices = neighbors.kneighbors(embeddings)  
    print('end of getting distances')
    
    max_dist = np.max(distances)

    filled_mat = lil_matrix((len(embeddings), len(embeddings)))

    if metric == 'euclidean':
        for i in range(len(embeddings)):
            filled_mat[i, indices[i]] = 1-(distances[i]/max_dist)
            filled_mat[indices[i], i] = 1-(distances[i]/max_dist)
    else:
        for i in tqdm(range(len(embeddings)), desc='filling similarity matrix'):
            filled_mat[i, indices[i]] = 1-distances[i]
            filled_mat[indices[i], i] = 1-distances[i]

    filled_mat = filled_mat.tocsc()

    return filled_mat

def get_similirity_matrix_compact(embeddings:pd.DataFrame, metric:str = 'euclidean', n_neighbors:int = 5) -> coex_matrix:
    """
    Function to calculate the similarity matrix between all words
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :return: coex_matrix - a compact matrix with the similarity score between all words
    """    
    print('fitting Nearest Neighbors')
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs = -1).fit(embeddings)
    print('End of fitting Nearest Neighbors')
    print('getting distances')
    distances, indices = neighbors.kneighbors(embeddings)  
    print('end of getting distances')

    max_dist = np.max(distances)

    unique_words = list(embeddings.index)
    dico = {i:dict() for i in unique_words}
  
    if metric == 'euclidean':
        for i in tqdm(range(len(unique_words)), desc='filling similarity matrix'):
            dico[unique_words[i]].update({unique_words[idx_word]:1-(dist/max_dist) for idx_word, dist in zip(indices[i], distances[i])})

            for j in list(dico[unique_words[i]].keys()):
                dico[j][unique_words[i]] = dico[unique_words[i]][j]

    else:
        for i in tqdm(range(len(unique_words)), desc='filling similarity matrix'):
            
            dico[unique_words[i]].update({unique_words[idx_word]:1-dist for idx_word, dist in zip(indices[i], distances[i])})
            for j in list(dico[unique_words[i]].keys()):
                if unique_words[i] != j:
                    dico[j][unique_words[i]] = dico[unique_words[i]][j]


    return coex_matrix(dico, unique_words)


def get_replaceable_words_end2end(corpus: Dict[int, str], embeddings:pd.DataFrame, thresh_prob: float = 0.0, metric:str = 'euclidean', n_neighbors:int = 5, alpha: float = 0.5, thresh: float = 0.8) -> dict[str, set[str]]:
    """
    Function to calculate the replaceable words in one function. It reduces the memory cost.
    :param corpus: dict[int, str] - a dictionary with the key being the document id and the value being the document text
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :param thresh_prob: float - coexistence probabilities that are below the threshold are considered as 0. Used to reduce memory consumption
    :param metric: str - metric of the similarity
    :param n_neighbors: int - number of neighbors to find
    """

    #ret_matrix = words_coexistence_probability_compact_parallel(corpus, thresh_prob)
    unique_words = get_unique_words(corpus)
    embeddings = embeddings.loc[list(unique_words)]
    ret_matrix = words_coexistence_probability_compact(corpus, list(embeddings.index), thresh_prob) * (1-alpha)

    print(ret_matrix[0,0])
    '''    
    words_in_common = list(set(ret_matrix.columns).intersection(set(embeddings.index)))
    embeddings = embeddings.loc[words_in_common]
    ret_matrix = ret_matrix.dico
    '''
    print('fitting Nearest Neighbors')
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs = -1).fit(embeddings)
    print('End of fitting Nearest Neighbors')
    print('getting distances')
    distances, indices = neighbors.kneighbors(embeddings)  
    print('end of getting distances')

    max_dist = np.max(distances)
    unique_words = list(embeddings.index)
    seen_couples = set()

    if metric == 'euclidean':
        for i in tqdm(range(len(unique_words)),desc='filling with similarity matrix'):
            ret_matrix[unique_words[i]] = {key:(1-alpha) * ret_matrix[unique_words[i]][key] for key in list(ret_matrix[unique_words[i]].keys())}

            for idx_word, dist in zip(indices[i], distances[i]):
                word = unique_words[idx_word]
                dist = (1-dist/max_dist) * alpha
                seen_couples.add((unique_words[i],word))

                if (word, unique_words[i]) in seen_couples:
                    if unique_words[i] in ret_matrix[word].keys():
                        ret_matrix[unique_words[i]][word] = ret_matrix[word][unique_words[i]]
                    else:
                        if word in ret_matrix[unique_words[i]]:
                            ret_matrix[unique_words[i]].pop(word)
                else:
                    if word in ret_matrix[unique_words[i]].keys():
                        ret_matrix[unique_words[i]][word] += dist
                        if ret_matrix[unique_words[i]][word] <= thresh:
                            ret_matrix[unique_words[i]].pop(word)
                        elif unique_words.index(word) < i:
                            ret_matrix[word][unique_words[i]] = ret_matrix[unique_words[i]][word]

                    else:
                        if dist > thresh:
                            ret_matrix[unique_words[i]][word] = dist
                            if unique_words.index(word) < i:
                                ret_matrix[word][unique_words[i]] = dist
    else:
        for i in tqdm(range(len(unique_words)),desc='filling with similarity matrix'):
            #print(ret_matrix[i].toarray())
        
            for idx_word, dist in zip(indices[i], distances[i]):
                
                dist = (1-dist) * alpha
                seen_couples.add((i,idx_word))

                if (idx_word, i) in seen_couples and idx_word != i :
                    ret_matrix[i, idx_word] = ret_matrix[idx_word, i]

                else:
                    if ret_matrix[i, idx_word] + dist > thresh:
                        ret_matrix[i, idx_word] += dist
                        if idx_word < i:
                            ret_matrix[idx_word, i] = ret_matrix[i, idx_word]
                    
    '''
    ret_matrix = {key:{key2:ret_matrix[key][key2] for key2 in list(ret_matrix[key].keys()) if ret_matrix[key][key2] > thresh} for key in tqdm(list(ret_matrix.keys()), desc='Finalizing the matrix')}


    
    ret_matrix_keys = list(ret_matrix.keys())
    for key in tqdm(ret_matrix_keys, desc = 'Finalizing replaceble words'):
        ret_matrix[key] = set(ret_matrix[key].keys())
    '''

    ret_matrix.data[ret_matrix.data <= thresh] = 0
    ret_matrix.eliminate_zeros()
    
    return ret_matrix















from multiprocessing import Pool
from tqdm import tqdm
from scipy.sparse import csc_matrix

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

def words_coexistence_probability_compact_parallel(corpus: dict[int, str], words: list[str], thresh_prob=0) -> csc_matrix:
    """
    Calculate the probability of coexistence of each pair of words in parallel.

    :param corpus: dict[int, str] - a dictionary with document IDs as keys and document texts as values.
    :param words: list[str] - a list of words for which to compute the probabilities.
    :param thresh_prob: float - the threshold probability below which pairs are ignored.
    :return: csc_matrix - the sparse matrix of coexistence probabilities.
    """
    # Build a mapping from each word to the set of document IDs in which it appears.
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
