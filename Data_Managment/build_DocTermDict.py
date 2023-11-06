import os, logging
import pandas as pd

from misc.dictionary_extension import DocumentTermDictionary
from misc.save_files import read_DocTermDict, write_DocTermDict
from misc.abb_med_disease import gen_abb_med_disease

def build_doc_term_dict(df: pd.DataFrame, config: dict, target_dir: str = None):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Doc Term Dict Generation - Started")

    # <editor-fold desc="Generate DocumentTermDictionary">
    text_column_dict = {(False, False, False): "clean_text",
                        (False, False, True): "disease_terms_clean_text",  (True, False, True): "disease_terms_clean_text",
                        (False, True, True): "disease_terms_clean_text", (True, True, True): "disease_terms_clean_text",
                        (False, True, False): "medical_terms_clean_text", (True, True, False): "abb_med_terms_clean_text",
                        (True, False, False): "abbreviations_clean_text",
                        }

    disease = False
    for med in [True, False]:
        for abb in [True, False]:
            abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
            if target_dir is not None:
                target_file = os.path.join(target_dir, f"{abb_med}", f"term_dictionary.dict")
            else:
                target_file = os.path.join(config["System"]["base_dir"], f"DocumentTermDictionary", f"{abb_med}", f"term_dictionary.dict")
            if os.path.isfile(target_file):
                continue
            term_dict = DocumentTermDictionary()
            [term_dict.add_to_dict(term=term, doc_id=id, year=year) for id, text, year in
             df[[text_column_dict.get((abb, med, disease), "clean_text"), "publication year"]].itertuples(index=True) for term in text]

            write_DocTermDict(target_dir=os.path.dirname(target_file),
                              obj=term_dict, logger=logger)
    disease = True
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    if target_dir is not None:
        target_file = os.path.join(target_dir, f"{abb_med}", f"term_dictionary.dict")
    else:
        target_file = os.path.join(config["System"]["base_dir"], f"DocumentTermDictionary", f"{abb_med}",
                                   f"term_dictionary.dict")
    if os.path.isfile(target_file):
        return
    term_dict = DocumentTermDictionary()
    [term_dict.add_to_dict(term=term, doc_id=id, year=year) for id, text, year in
     df[[text_column_dict.get((abb, med, disease), "clean_text"), "publication year"]].itertuples(index=True) for
     term in text]

    write_DocTermDict(target_dir=os.path.dirname(target_file),
                      obj=term_dict, logger=logger)
    # </editor-fold>


# Loads TermDict
def add_publications_doc_term_dict(df: pd.DataFrame, config: dict, target_dir: str):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Doc Term Dict Update - Started")
    if target_dir is not None:
        target_file = os.path.join(target_dir, f"term_dictionary.dict")
    else:
        target_file = os.path.join(config["System"]["base_dir"], f"DocumentTermDictionary", f"term_dictionary.dict")


    # <editor-fold desc="Updateing DocumentTermDictionary">
    term_dict = read_DocTermDict(target_dir=config["System"]["base_dir"],
                                 logger=logger)
    [term_dict.add_to_dict(term=term, doc_id=id, year=year) for id, text, year in
     df[["clean_text", "publication year"]].itertuples(index=True) for term in text]
    write_DocTermDict(target_dir=target_file,
                      obj=term_dict, logger=logger)
    logger.info("Doc Term Dict Update - Finished")
    # </editor-fold>

# Takes TermDict Preloaded
def add_publications_doc_term_dict_instance(df: pd.DataFrame, term_dict: DocumentTermDictionary, path: str, config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Doc Term Dict Update - Started")

    # <editor-fold desc="Updateing DocumentTermDictionary">
    [term_dict.add_to_dict(term=term, doc_id=id, year=year) for id, text, year in
     df[["clean_text", "publication year"]].itertuples(index=True) for term in text]
    if path is not None:
        write_DocTermDict(target_dir=config["System"]["base_dir"],
                          obj=term_dict, logger=logger)
        logger.info("Doc Term Dict Update - Finished")
    # </editor-fold>
    return term_dict


# Takes TermDict Preloaded
def update_doc_term_dict_instance(term_dict: DocumentTermDictionary, term_dict_addition: DocumentTermDictionary, config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Doc Term Dict Update - Started")

    term_dict.term2key_doc.update(term_dict_addition.term2key_doc)
    term_dict.key2term.update(term_dict_addition.key2term)
    return term_dict


# Takes TermDict Preloaded
def select_terms_to_keep_doc_term_dict_instance(term_dict: DocumentTermDictionary, terms_to_keep: list, config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Doc Term Dict Update - Started")
    if terms_to_keep is None:
        terms_to_keep = config["Query"]["terms_to_keep_in_embedding"]

    term_dict.term2key_doc = {term: term_dict.term2key_doc.get(term, None) for term in terms_to_keep}
    return term_dict

def get_earliest_meantion_of_term(term: str, term_dict: DocumentTermDictionary):
    key_docs = term_dict.term2key_doc.get(term, (-1, []))
    years = [term_dict.doc2year.get(i, None) for i in key_docs[1] if term_dict.doc2year.get(i, None) is not None]
    if len(years) != 0:
        return min(years)
    return None
