import pandas as pd 
from tqdm import tqdm
from utils_func import matrix_creation

import multiprocessing as mp
from typing import Dict, Set, Any

import fasttext

class Graph:
    def __init__(self, graph_dict=None):
        if graph_dict is None:
            graph_dict = {}
        self.graph_dict = graph_dict

    def vertices(self):
        return list(self.graph_dict.keys())

    def edges(self):
        return self.generate_edges()

    def add_vertex(self, vertex):
        if vertex not in self.graph_dict:
            self.graph_dict[vertex] = []

    def add_edge(self, edge):
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.graph_dict:
            self.graph_dict[vertex1].append(vertex2)
        else:
            self.graph_dict[vertex1] = [vertex2]

    def generate_edges(self):
        edges = []
        for vertex in self.graph_dict:
            for neighbour in self.graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def find_path(self, start_vertex, end_vertex, path=None):
        if path is None:
            path = []
        graph = self.graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path:
                    return extended_path
        return None

    def find_all_paths(self, start_vertex, end_vertex, path=None):
        if path is None:
            path = []
        graph = self.graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths:
                    paths.append(p)
        return paths
    
    def dfs(self, start_vertex):
        explored = set()
        frontier = [start_vertex]

        while len(frontier) > 0:
            vertex = frontier.pop()
            if vertex not in explored:
                explored.add(vertex)
                frontier.extend(self.graph_dict[vertex] - explored)
        return explored
    
    def find_all_cycles(self):
        all_cycles = []
        vertices = set(self.vertices())

        while vertices:
            vertex = vertices.pop()
            cycle = self.dfs(vertex)
            vertices -= cycle
            all_cycles.append(cycle)
        return all_cycles
    
    def __str__(self):
        res = "vertices: "
        for k in self.graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.generate_edges():
            res += str(edge) + " "
        return res

'''
def get_replaceable_words(similarity_matrix:pd.DataFrame, coexistence_matrix:pd.DataFrame, alpha = 0.5, thresh = 0.8) -> dict[str, set[str]]:
    """
    Get for each word, the set of words that can replace it in a sentence according to the constraints on similarity and coexistence matrix.\n
    :param similarity_matrix: A pandas DataFrame containing the similarity matrix between words.\n
    :param coexistence_matrix: A pandas DataFrame containing the coexistence matrix between words.\n
    :param alpha: A float between 0 and 1, the weight of the similarity matrix in the final decision.\n
    :param thresh: A float between 0 and 1, the threshold to consider a word as a possible replacement.\n
    :return: A dictionary containing for each word, the set of words that can replace it in a sentence.
    """
    all_words = list(set(similarity_matrix.index).intersection(set(coexistence_matrix.index)))
    to_ret = {}
    if type(coexistence_matrix) == matrix_creation.coex_matrix:
        for word in tqdm(all_words):
            temp = similarity_matrix.loc[word].multiply(alpha).add(coexistence_matrix.loc(word).multiply(1-alpha), fill_value=0)
            to_ret[word] = set(temp[temp > thresh].index)
    else:
        for word in tqdm(all_words):
            temp = alpha * similarity_matrix.loc[word] + (1-alpha) * coexistence_matrix.loc[word]
            to_ret[word] = set(temp[temp > thresh].index)
            
    return to_ret
'''
def clusters_dict(clusters:list[set[str]]) -> dict[str:str]:
    """
    Get a dictionary containing for each word, the cluster it belongs to.\n
    :param clusters: A list of sets containing the clusters of words.\n
    :return: A dictionary containing for each word, the cluster it belongs to.
    """
    to_ret = {}
    unique_words = {word for cluster in clusters for word in cluster}
    clust_names = list(range(len(clusters)))
    for cluster  in tqdm(clusters, desc= 'Creating clusters dict'):
        for word in cluster:
            to_ret[word] = clust_names[clusters.index(cluster)]
    return to_ret

def rewrite_text(text:str, clust_dict:dict[str,str], fasttext_model: fasttext.FastText = None, thresh=0.75) -> str:
    """
    Rewrite the text using the clusters dictionary.\n
    :param text: The text to rewrite.\n
    :param clust_dict: The dictionary containing for each word, the cluster it belongs to.\n
    :return: The rewritten text.
    """
    text = text.split()
    words = set(clust_dict.keys())
    l_words = len(words)
    to_ret = []
    for i in range(len(text)):
        words.add(text[i])
        if len(words) == l_words:
            to_ret.append(str(clust_dict[text[i]]))
        else:
            words = set(clust_dict.keys())
            if fasttext_model is not None:
                neighs = fasttext_model.get_nearest_neighbors(text[i], k=500)
                neighs = [(i[0], clust_dict[i[1]]) for i in neighs if i[1] in clust_dict.keys() and i[0]>thresh]
                diff_clust = [i[1] for i in neighs]
       
                if neighs != []:
                    dico_clust = dict()
                    for sim, clust in neighs:
                        if clust in dico_clust.keys():
                            dico_clust[clust] += sim
                        else: 
                            dico_clust[clust] = sim
                    
                    dico_clust = {key:dico_clust[key]/diff_clust.count(key) for key in list(dico_clust.keys())}
                    best_key = list(dico_clust.keys())[0]
                    for key in list(dico_clust.keys()):
                        if dico_clust[key] > dico_clust[best_key]:
                            best_key = key
                    
                    to_ret.append(str(best_key))

    return ' '.join(to_ret)

def rewrite_corpus(corpus:dict[str,str], clust_dict:dict[str,str]) -> dict[str,str]:
    """
    Rewrite the corpus using the clusters dictionary.\n
    :param corpus: The corpus to rewrite.\n
    :param clust_dict: The dictionary containing for each word, the cluster it belongs to.\n
    :return: The rewritten corpus.
    """
    to_ret = {key:rewrite_text(corpus[key], clust_dict) for key in tqdm(corpus.keys(), desc='Rewriting corpus')}

    return to_ret





###################################################################################################################
# Parallelization

import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from utils_func import matrix_creation  # Adjust import as needed

global_similarity_matrix = None
global_coex_matrix = None
global_alpha = None
global_thresh = None

def init_worker(sim_mat: pd.DataFrame, coex_mat: pd.DataFrame, alpha: float, thresh: float):
    """
    Initializer for worker processes.
    Sets up global variables for use in each worker.
    """
    global global_similarity_matrix, global_coex_matrix, global_alpha, global_thresh
    global_similarity_matrix = sim_mat
    global_coex_matrix = coex_mat
    global_alpha = alpha
    global_thresh = thresh

def worker_replaceable(word: str) -> tuple[str, set[str]]:
    """
    Worker function that computes the set of words that can replace a given word.
    It uses the global matrices and parameters.
    """
    # Check which method to use based on the type of coexistence matrix.
    # (Assumes matrix_creation.coex_matrix is a custom type.)
    if isinstance(global_coex_matrix, matrix_creation.coex_matrix) and isinstance(global_similarity_matrix, matrix_creation.coex_matrix):
        # Use pandas Series.multiply and DataFrame.add methods.
        # Note: We use .loc[word] for both matrices.
        temp = global_similarity_matrix.loc(word).multiply(global_alpha).add(
            global_coex_matrix.loc(word).multiply(1 - global_alpha), fill_value=0
        )
    elif isinstance(global_similarity_matrix, matrix_creation.coex_matrix):
        # Use pandas Series.multiply and DataFrame.add methods.
        # Note: We use .loc[word] for both matrices.
        temp = global_similarity_matrix.loc(word).multiply(global_alpha).add(
            global_coex_matrix.loc[word].multiply(1 - global_alpha), fill_value=0
        )
    elif isinstance(global_coex_matrix, matrix_creation.coex_matrix):
        # Use pandas Series.multiply and DataFrame.add methods.
        # Note: We use .loc[word] for both matrices.
        temp = global_similarity_matrix.loc[word].multiply(global_alpha).add(
            global_coex_matrix.loc(word).multiply(1 - global_alpha), fill_value=0
        )
    else:
        # Use vectorized arithmetic
        temp = global_alpha * global_similarity_matrix.loc[word] + (1 - global_alpha) * global_coex_matrix.loc[word]
    
    # Select words that exceed the threshold.
    replaceable = set(temp[temp > global_thresh].index)
    return word, replaceable

def get_replaceable_words(similarity_matrix: pd.DataFrame, 
                          coexistence_matrix: pd.DataFrame, 
                          alpha: float = 0.5, 
                          thresh: float = 0.8) -> dict[str, set[str]]:
    """
    For each word, computes the set of words that can replace it based on a weighted 
    combination of similarity and coexistence scores.
    
    The computation is parallelized across words.
    
    :param similarity_matrix: DataFrame with similarity scores (rows and columns are words)
    :param coexistence_matrix: DataFrame with coexistence scores (rows and columns are words)
    :param alpha: Weight for similarity (between 0 and 1)
    :param thresh: Threshold above which a word is considered a replacement
    :return: Dictionary mapping each word to a set of replacement words.
    """
    # Compute the list of words that are common to both matrices.
    all_words = list(set(similarity_matrix.index).intersection(set(coexistence_matrix.index)))
    results = {}

    # Create a pool of worker processes.
    # Be sure to protect the entry point with if __name__ == '__main__' when needed.
    with Pool(initializer=init_worker, initargs=(similarity_matrix, coexistence_matrix, alpha, thresh)) as pool:
        # Use pool.imap to distribute work and wrap it in a tqdm progress bar.
        for word, replacements in tqdm(pool.imap(worker_replaceable, all_words),
                                       total=len(all_words),
                                       desc="Processing words"):
            results[word] = replacements

    return results