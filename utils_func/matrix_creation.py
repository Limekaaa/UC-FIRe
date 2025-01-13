import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


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

