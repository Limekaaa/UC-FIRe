import pandas as pd 
from tqdm import tqdm

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
    for word in tqdm(all_words):
        temp = alpha * similarity_matrix.loc[word] + (1-alpha) * coexistence_matrix.loc[word]
        to_ret[word] = set(temp[temp > thresh].index)

    return to_ret

def clusters_dict(clusters:list[set[str]]) -> dict[str:str]:
    """
    Get a dictionary containing for each word, the cluster it belongs to.\n
    :param clusters: A list of sets containing the clusters of words.\n
    :return: A dictionary containing for each word, the cluster it belongs to.
    """
    to_ret = {}
    unique_words = {word for cluster in clusters for word in cluster}
    clust_names = list(range(len(clusters)))
    for cluster  in clusters:
        for word in cluster:
            to_ret[word] = clust_names[clusters.index(cluster)]
    return to_ret

def rewrite_text(text:str, clust_dict:dict[str,str]) -> str:
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
    return ' '.join(to_ret)

def rewrite_corpus(corpus:dict[str,str], clust_dict:dict[str,str]) -> dict[str,str]:
    """
    Rewrite the corpus using the clusters dictionary.\n
    :param corpus: The corpus to rewrite.\n
    :param clust_dict: The dictionary containing for each word, the cluster it belongs to.\n
    :return: The rewritten corpus.
    """
    to_ret = {key:rewrite_text(corpus[key], clust_dict) for key in tqdm(corpus.keys())}

    return to_ret