from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from utils_func import corpus_processing, matrix_creation

from typing import Literal
from tqdm import tqdm
import pandas as pd

from multiprocessing import Pool, cpu_count
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import tempfile
import fasttext
import numpy as np

import tempfile

def create_vectors(corpus:dict[str:str], dim:int, path_to_save_vectors:str, path_to_save_model:str = '', epochs:int=5, model:Literal['skipgram', 'cbow'] = 'skipgram') -> pd.DataFrame:

    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        for keys in tqdm(corpus.keys(), desc="Creating file to train fasttext model"):
            temp_file.write(f"{corpus[keys]}\n")
        temp_file_path = temp_file.name

    model = fasttext.train_unsupervised(temp_file_path, model=model, epoch = epochs, dim=dim)
    unique_words = list(matrix_creation.get_unique_words(corpus))

    word_vectors = np.array([model.get_word_vector(word) for word in unique_words])

    embeddings = pd.DataFrame(word_vectors, index=unique_words)
    embeddings.to_csv(path_to_save_vectors, sep = ' ')
    if path_to_save_model != '':
        model.save_model(path_to_save_model)

    return embeddings

