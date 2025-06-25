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
import numpy as np

import tempfile

import torch

from transformers import BertTokenizer, BertModel

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from collections import defaultdict



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

def split_text(text, max_len=512, context_window=128, device = "cpu"):
    n = len(text)
    if n <= max_len:
        return [text.to(device)]
    else:
        return [text[i:i+max_len].to(device) for i in range(0, n, max_len-context_window)]

def create_vectors(corpus:dict[str:str], model, tokenizer, device = 'cpu',path_to_save_vectors:str=None):
    max_len = model.config.max_position_embeddings
    word_embeddings = defaultdict(list)
    for key in corpus.keys():
        tokens = tokenizer(corpus[key], return_tensors="pt")#, padding=True, truncation=True)
        
        input_ids_split = split_text(tokens['input_ids'][0], max_len, device=device)
        token_type_ids_split = split_text(tokens['token_type_ids'][0], max_len, device=device)
        attention_mask_split = split_text(tokens['attention_mask'][0],max_len, device=device)
        n = len(input_ids_split)
        all_vectors = []
        all_input_ids = []
        for j in range(n):
            to_add = {}
            to_add['input_ids'] = input_ids_split[j].clone().detach().unsqueeze(0).to(device)
            to_add['token_type_ids'] = token_type_ids_split[j].clone().detach().unsqueeze(0).to(device)
            to_add['attention_mask'] = attention_mask_split[j].clone().detach().unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(**to_add)
            
            hidden_states = outputs.hidden_states  # All layers

            # Compute the mean of the last 4 layers
            last_4_layers = torch.stack(hidden_states[-4:])  # Shape: (4, batch, seq_len, hidden_dim)
            mean_last_4 = last_4_layers.mean(dim=0).squeeze(0)  # Shape: (seq_len, hidden_dim)

            all_vectors.append(mean_last_4.detach().cpu().numpy())
        # Map tokens back to words
        mean_last_4 = np.concatenate(all_vectors, axis=0)
        tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())

        for i, token in enumerate(tokenized_text):
            #if token.startswith("##"):  # Skip subwords to get only whole word representations
            #    continue
            word_embeddings[token].append(mean_last_4[i])

    # Aggregate embeddings per word (mean over occurrences)
    final_word_embeddings = {word: np.mean(vectors, axis=0) for word, vectors in word_embeddings.items()}
    embeddings = pd.DataFrame(final_word_embeddings).T
    if path_to_save_vectors != None:
        embeddings.to_csv(path_to_save_vectors, sep=' ')
    return embeddings