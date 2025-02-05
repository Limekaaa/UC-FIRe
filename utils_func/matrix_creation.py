import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import multiprocessing as mp
from typing import Dict, Set, Any

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

def words_coexistence_probability_compact(corpus:dict[int, str], thresh_prob = 0) -> coex_matrix:
    """
    Function to calculate the probability of coexistence of each pair of words in the corpus in a compact way
    :param corpus: dict[int, str] - a dictionary with the key being the document id and the value being the document text
    :param thresh_prob: float - the threshold probability to consider the coexistence of two words, allow to reduce the size of the matrix
    :return: pd.DataFrame - a dataframe with the probability of coexistence of each pair of words in the corpus
    """
    dico = dict()
    word_presence = get_word_presence(corpus)
    unique_words = list(get_unique_words(corpus))

    for word2 in tqdm(range(len(unique_words))):
        dico[unique_words[word2]] = dict()
        for word1 in range(len(unique_words)):
            inter = len(word_presence[unique_words[word1]].intersection(word_presence[unique_words[word2]]))
            if inter > 0:
                prob = inter/max(len(word_presence[unique_words[word1]] | word_presence[unique_words[word2]]), 1)
                if prob > thresh_prob:
                    dico[unique_words[word2]][unique_words[word1]] = prob
    return coex_matrix(dico, unique_words)


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
                                                  thresh_prob: float = 0) -> pd.DataFrame:
    """
    Parallelized function to calculate the probability of coexistence of each pair of words.
    """
    # Precompute necessary data structures
    word_presence = matrix_creation.get_word_presence(corpus)
    unique_words = list(matrix_creation.get_unique_words(corpus))
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
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(embeddings)
    return neighbors.kneighbors(embeddings.loc[word].values.reshape(1,-1))


def get_similarity_matrix(embeddings:pd.DataFrame, metric:str = 'euclidean', n_neighbors:int = 5) -> pd.DataFrame:
    """
    Function to calculate the similarity matrix between all words
    :param embeddings: pd.DataFrame - a dataframe with the embeddings of each word
    :return: pd.DataFrame - a dataframe with the similarity score between all words
    """

    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(embeddings)
    distances, indices = neighbors.kneighbors(embeddings) # 
    max_dist = np.max(distances)
    filled_mat= np.zeros((len(embeddings), len(embeddings)))

    if metric == 'euclidean':
        for i in range(len(embeddings)):
            filled_mat[i, indices[i]] = 1-(distances[i]/max_dist)
            filled_mat[indices[i], i] = 1-(distances[i]/max_dist)
    else:
        for i in range(len(embeddings)):
            filled_mat[i, indices[i]] = 1-distances[i]
            filled_mat[indices[i], i] = 1-distances[i]

    similarity_matrix = pd.DataFrame(filled_mat, index=embeddings.index, columns=embeddings.index)

    return similarity_matrix

