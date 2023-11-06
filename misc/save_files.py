import os, pickle, logging
import pandas as pd
from gensim.models.word2vec import Word2Vec, Word2VecKeyedVectors
from gensim.models import KeyedVectors
from misc.dictionary_extension import DocumentTermDictionary



def read_pickle(path: str, logger: logging.Logger):
    if not os.path.isfile(path=path):
        logger.error(f"{path} - does not exist")
        return None
    with open(path, "rb") as f:
        obj = pickle.load(file=f)
    logger.debug(f"loaded object from {path}")
    return obj


def write_pickle(path: str, obj, logger: logging.Logger):
    if os.path.isfile(path=path):
        logger.debug(f"{path} - was overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj=obj, file=f)
    logger.debug(f"saved object to {path}")
    

def read_pandas_from_csv(path: str, logger: logging.Logger, index_col: bool =True, **kwargs):
    if not os.path.isfile(path=path):
        logger.error(f"{path} - does not exist")
        return None
    if not index_col:
        df = pd.read_csv(path, **kwargs)
    else:
        df = pd.read_csv(path, index_col=[0], **kwargs)

    logger.debug(f"loaded object from {path}")
    return df


def write_pandas_to_csv(path: str, df: pd.DataFrame, logger: logging.Logger):
    if os.path.isfile(path=path):
        logger.debug(f"{path} - was overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    logger.debug(f"saved object to {path}")

def write_w2v_model(path: str, obj, logger: logging.Logger):
    if os.path.isfile(path=path):
        logger.debug(f"{path} - was overwritten")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if type(obj) == Word2Vec:
        obj.wv.save(path)
    elif type(obj) == Word2VecKeyedVectors:
        obj.save(path)
    else:
        logger.error(f"Could not save obj of type {type(obj)}")
        return
    logger.debug(f"saved W2V Model to {path}")

def read_w2v_model(path: str, logger: logging.Logger):

    if not os.path.isfile(path=path):
        logger.debug(f"{path} - does not exist")
        return None

    model = KeyedVectors.load(path, mmap="r")
    logger.debug(f"loaded object from {path}")
    return model


def write_DocTermDict(target_dir: str, obj: DocumentTermDictionary, logger: logging.Logger):
    path = os.path.join(target_dir, f"term_dictionary.dict")
    if os.path.isfile(path=path):
        logger.debug(f"{path} - was overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    obj.save(path=path)

    logger.debug(f"saved DocumentTermDictionary to {path}")

def read_DocTermDict(target_dir: str, logger: logging.Logger):
    path = os.path.join(target_dir, f"term_dictionary.dict")
    if not os.path.isfile(path=path):
        logger.debug(f"{path} - does not exist")
        return None

    term_dict = DocumentTermDictionary().load(path=path)
    logger.debug(f"loaded DocumentTermDictionary from {path}")
    return term_dict