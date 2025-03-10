import spacy
from tqdm import tqdm
import pandas as pd 

from multiprocessing import Pool, cpu_count
import multiprocessing
import re

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
clean_tokens = lambda tokens : ' '.join([token.lemma_.lower() for token in tokens if token.lemma_.lower() not in stopwords and not token.is_punct])

splitter = re.compile(r"(?u)\b\w+\b").findall

def pre_process(elem_to_preprocess: tuple[int, dict[str,str]]) -> tuple[int, str]:
  """
  Preprocesses the text data in the corpus\n
  :param elem_to_preprocess: Tuple containing the key and the value of the element to preprocess\n
  :return: Tuple containing the key and the preprocessed text data\n
  """

  key, val = elem_to_preprocess
  return key, f"{clean_tokens(nlp(val['title'].lower()))} {clean_tokens(nlp(val['text'].lower()))}" # Cleaning the text document
 
# Helper function to process a single key-value pair
def process_item(item):
    key, value = item
    return key, pre_process((key, value))[1]
'''
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
'''
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
    
    
def init_worker():
    """
    Cette fonction d'initialisation est appelée une fois 
    dans chaque processus fils, juste après sa création.
    """
    global nlp, stopwords
    nlp = spacy.load('en_core_web_sm')
    stopwords = nlp.Defaults.stop_words

def clean_tokens(tokens):
    """ 
    Nettoie les tokens en supprimant les stopwords et la ponctuation,
    puis renvoie la lemmatisation en minuscules.
    """
    return ' '.join([
        token.lemma_.lower()
        for token in tokens
        if (token.lemma_.lower() not in stopwords and not token.is_punct)
    ])

'''
def preprocess_text(text):
    # Add spaces around specific punctuation: . ! ? , ' / ( )
    text = re.sub(r"([.\!?,'/()])", r" \1 ", text)

    # Add spaces around underscores (_), but not within numbers (e.g., var_1 remains unchanged)
    text = re.sub(r"(?<!\d)_(?!\d)", " _ ", text)

    # Add a space after opening curly braces if followed by text (for LaTeX cases like \text{word})
    text = re.sub(r"(\{)([^\s])", r"\1 \2", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
'''
def preprocess_text(text):
    # Add spaces around specific punctuation: . ! ? , ' / ( )
    # But avoid adding spaces to dots in decimal numbers (e.g., 0.75)
    text = re.sub(r"(?<!\d)([.\!?,'/()])(?!\d)", r" \1 ", text)

    # Add spaces around underscores (_), but not within numbers (e.g., var_1 remains unchanged)
    text = re.sub(r"(?<!\d)_(?!\d)", " _ ", text)

    # Add spaces before `[`
    text = re.sub(r"(?<=\w|\.)\[", " [", text)
    
    # Add spaces after `]` if followed by a word
    text = re.sub(r"\](?=\w)", "] ", text)

    # Add a space after opening curly braces if followed by text (for LaTeX cases like \text{word})
    text = re.sub(r"(\{)([^\s])", r"\1 \2", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def process_item(item):
    """
    Fonction qui réalise le prétraitement d'un seul élément du corpus.
    Retourne un tuple (key, texte_nettoyé).
    """
    key, val = item
    
    #title_clean = clean_tokens(nlp(val['title'].lower()))
    #text_clean  = clean_tokens(nlp(val['text'].lower()))

    title_clean = clean_tokens(nlp(preprocess_text(val['title'])))
    text_clean  = clean_tokens(nlp(preprocess_text(val['text'])))

    return key, f"{title_clean} {text_clean}"

def preprocess_corpus_dict(corpus):
    """
    Prétraite l'ensemble du corpus (dict) en parallèle,
    et affiche la progression via tqdm.
    """
    items = list(corpus.items())
    results_dict = {}

    # Pool de workers = nbre de CPUs dispo, 
    # avec init_worker pour charger spaCy dans chaque sous-processus
    with Pool(processes=cpu_count(), initializer=init_worker) as pool:
        # imap_unordered retourne les résultats au fil de l'eau (ordre non garanti)
        # On itère dessus pour mettre à jour la barre de progression
        for key, processed_text in tqdm(
            pool.imap_unordered(process_item, items),
            total=len(items),
            desc="Prétraitement du corpus"
        ):
            results_dict[key] = processed_text

    return results_dict
