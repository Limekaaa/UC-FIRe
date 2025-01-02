import spacy
from tqdm import tqdm
import pandas as pd 

from multiprocessing import Pool, cpu_count


nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
clean_tokens = lambda tokens : ' '.join([token.lemma_.lower() for token in tokens if token not in stopwords and not token.is_punct])


def pre_process(elem_to_preprocess: tuple[int, dict[str,str]]) -> tuple[int, str]:
  """
  Preprocesses the text data in the corpus\n
  :param elem_to_preprocess: Tuple containing the key and the value of the element to preprocess\n
  :return: Tuple containing the key and the preprocessed text data\n
  """

  key, val = elem_to_preprocess
  return key, f"{clean_tokens(nlp(val['title']))} {clean_tokens(nlp(val['text']))}" # Cleaning the text document
 
# Helper function to process a single key-value pair
def process_item(item):
    key, value = item
    return key, pre_process((key, value))[1]

def preprocess_corpus_dict(corpus: dict[int, dict[str, str]]) -> dict[int, str]:
    """
    Preprocesses the text data in the corpus in parallel\n
    :param corpus: The corpus to preprocess
    :return: The preprocessed corpus
    """
    # Convert the corpus to a list of items for parallel processing
    items = list(corpus.items())

    # Use a multiprocessing pool for parallel execution
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_item, items), total=len(items)))

    # Combine results into a dictionary
    cleaned_corpus = {key: value for key, value in results}

    return cleaned_corpus

""" Non parallel version

def preprocess_corpus_dict(corpus: dict[int, dict[str,str]]) -> dict[int, str]:
    '''
    Preprocesses the text data in the corpus\n
    :param corpus: The corpus to preprocess\n
    :return: The preprocessed corpus\n
    '''
    cleaned_corpus = {}
    for key in tqdm(corpus.keys()):
        cleaned_corpus[key] = pre_process((key, corpus[key]))[1]
    return cleaned_corpus
"""

def save_processed_corpus(corpus, path_to_save):
    """
    Saves the preprocessed corpus to a csv file\n
    :param corpus: The preprocessed corpus\n
    :param path_to_save: The path to save the preprocessed corpus\n
    """
    
    df = pd.DataFrame.from_dict(corpus, orient='index')
    df.reset_index(inplace=True)
    df.columns = ["doc_id", "text"]
    df.to_csv(path_to_save, index=False)