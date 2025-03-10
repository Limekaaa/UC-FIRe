import spacy
from tqdm import tqdm
import pandas as pd 

from multiprocessing import Pool, cpu_count
import multiprocessing
import re

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
 
def save_processed_corpus(corpus:dict[str:str], path_to_save:str):
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


def preprocess_text(text:str) -> str:
    """
    Preprocess a text by applying the following steps:
    - Remove extra whitespaces
    - Add spaces around specific punctuation
    - Add spaces around underscores (_) but not within numbers
    - Add a space after opening curly braces if followed by text
    - Convert to lowercase
    - Normalize multiple spaces
    """
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
    Function that pre-processes a single element of the corpus.
    Returns a tuple (key, cleaned_text).
    """
    key, val = item
    
    #title_clean = clean_tokens(nlp(val['title'].lower()))
    #text_clean  = clean_tokens(nlp(val['text'].lower()))

    title_clean = clean_tokens(nlp(preprocess_text(val['title'])))
    text_clean  = clean_tokens(nlp(preprocess_text(val['text'])))

    return key, f"{title_clean} {text_clean}"

def preprocess_corpus_dict(corpus: dict[str, dict[str, str]]) -> dict[str, str]:
    """
    Preprocesses a corpus in parallel.
    Returns a dictionary with the same keys as the input,
    and the cleaned text as values.
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
