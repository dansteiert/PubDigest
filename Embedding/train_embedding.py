import os, datetime, logging, copy
import pandas as pd
from gensim.models import word2vec, KeyedVectors, Word2Vec

from misc.save_files import write_w2v_model, read_DocTermDict, read_w2v_model
from misc.abb_med_disease import gen_abb_med_disease
from misc.dictionary_extension import DocumentTermDictionary

from Data_Managment.update_database import data_collection
from Data_Managment.build_DocTermDict import build_doc_term_dict


def train_w2v_model(config: dict, df: pd.DataFrame, target_dir: str, all_lower_case: bool = False,
                    term_dict_dir: str = None):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    logger.info("Start generating w2v models")

    year_list = list(reversed(range(datetime.date.today().year, config["NLP"]["min_year"] - 1,
                                    -config["NLP"]["year_interval"])))

    # <editor-fold desc="Train models">
    logger.info(f"Train models for years: {year_list}")
    model_dict = {}
    for year in year_list:
        target_file = os.path.join(target_dir, f"w2v_{year}.kv")
        if not os.path.isfile(target_file):
            logger.info(f"Train models for year: {year}")
            text = [i for i in df[df["publication year"] <= year]["clean_text"].tolist() if i is not None]
            if len(text) == 0:
                continue
            if all_lower_case:
                text = [[j.lower() for j in i] for i in text]
            model = word2vec.Word2Vec(sentences=text, workers=config["System"]["processors"],
                                      sg=0 if config["NLP"]["skip_gram_CBOW"] == "CBOW" else 1, **config["NLP"]["word2vec"])
            write_w2v_model(path=target_file, obj=model, logger=logger)
            model_dict[year] = model
        else:
            model_dict[year] = read_w2v_model(path=target_file, logger=logger)

        # <editor-fold desc="Generate Special Term specific embeddings">
        disease = False
        for med in [True, False]:
            for abb in [True, False]:
                gen_abb_med_models(config=config, model=model_dict[year], abb=abb, med=med, disease=disease,
                                   file_name=f"w2v_{year}.kv",
                                   terms_to_keep_set=set([*config["Query"]["terms_to_keep_in_embedding"], config["Query"]["prediction_term"]]),
                                   target_dir=target_dir, term_dict_dir=term_dict_dir)
        disease = True
        gen_abb_med_models(config=config, model=model_dict[year], abb=abb, med=med, disease=disease,
                           file_name=f"w2v_{year}.kv", target_dir=target_dir,
                           terms_to_keep_set=set([*config["Query"]["terms_to_keep_in_embedding"], config["Query"]["prediction_term"]]),
                           term_dict_dir=term_dict_dir)
        # </editor-fold>
    # </editor-fold>
    return model_dict
    # </editor-fold>


def train_joined_w2v_model(config: dict, df_publication_base: pd.DataFrame, df_publication_query: pd.DataFrame,
                           target_dir: str, term_dict_dir: str, all_lower_case: bool=False):
    """
    A Wrapper, which first joins two datasets and trains an embedding on this dataset
    """

    columns = ["clean_text", "disease_terms_clean_text", "medical_terms_clean_text", "abb_med_terms_clean_text",
               "abbreviations_clean_text", "publication year"]
    df = pd.concat([df_publication_base[columns],
                    df_publication_query[columns]])

    build_doc_term_dict(df=df, config=config, target_dir=target_dir)

    train_w2v_model(config=config, df=df, target_dir=target_dir, all_lower_case=all_lower_case, term_dict_dir=term_dict_dir)


def embedding_wrapper_multi_dataset(config: dict, df_publications: pd.DataFrame):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    for term, full_query in zip(config['Query']["alternate_query_terms"],
                                config['Query']["alternate_query_parameters"]["full_query"]):

        config["System"]["base_dir"] = os.path.join(config["System"]["working_dir"], "data", term)
        logger.info(f"Train Embedding for - {config['Query']['base_query_term']} & {term}")

        df_publications_query = data_collection(config=config, full_query=full_query)
        build_doc_term_dict(df=df_publications_query, config=config)

        target_dir = os.path.join(config["System"]["working_dir"], "data",
                                                       config["Query"]["base_query_term"],
                                                       config["NLP"]["skip_gram_CBOW"], term)

        train_joined_w2v_model(config=config, df_publication_base=df_publications,
                               df_publication_query=df_publications_query,
                               target_dir=target_dir,
                               term_dict_dir=target_dir)

    config["System"]["base_dir"] = os.path.join(config["System"]["working_dir"], "data",
                                                config["Query"]["base_query_term"])


def gen_abb_med_models(config: dict, model: KeyedVectors, abb: bool, med: bool, disease: bool, file_name: str,
                       terms_to_keep_set: set, target_dir: str, term_dict_dir: str):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    if (not abb and not med and not disease):
        return
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

    target_file = os.path.join(target_dir, abb_med, file_name)
    if os.path.isfile(target_file):
        return

    if term_dict_dir is not None:
        term_dict = read_DocTermDict(
            target_dir=os.path.join(term_dict_dir, f"{abb_med}"), logger=logger)
    else:
        term_dict = read_DocTermDict(
            target_dir=os.path.join(config["System"]["base_dir"], f"DocumentTermDictionary", f"{abb_med}"), logger=logger)

    if type(model) == Word2Vec:
        model = model.wv

    terms_to_keep = {val: index for index, val in enumerate(model.index2word) if
                     term_in_abb_med(val, term_dict=term_dict, terms_to_keep=terms_to_keep_set)}

    # <editor-fold desc="Copy Model and Select Desired Terms">
    copy_model = copy.deepcopy(model)
    copy_model.vectors = copy_model.vectors[list(terms_to_keep.values())]
    copy_model.index2word = [val for i, val in enumerate(copy_model.index2word) if i in terms_to_keep.values()]
    copy_model.vocab = {k: v for k, v in copy_model.vocab.items() if terms_to_keep.get(k, False)}

    [setattr(copy_model.vocab[k], "index", index) for index, k in enumerate(terms_to_keep.keys())]
    write_w2v_model(
        path=target_file,
        obj=copy_model, logger=logger)


def term_in_abb_med(term: str, term_dict: DocumentTermDictionary, terms_to_keep: set):
    if term in terms_to_keep:
        return True
    if term in term_dict.term2key_doc:
        return True
    return False
