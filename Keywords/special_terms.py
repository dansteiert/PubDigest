import os, sys, re, enchant, logging
sys.path.append(os.getcwd())

import nltk
try:
    from nltk.tokenize import TweetTokenizer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    from nltk.util import ngrams
except:
    nltk.download()


import pandas as pd
import numpy as np

from misc.save_files import write_pandas_to_csv, read_pandas_from_csv, write_pickle, read_pickle
from misc.counter import count_dict_three_level


def extract_medical_terms(document_text_list: list, config: dict):
    """
    :param document_text_list: list -> corpus_list[document_1_list[term_1, term_2, ... term_n], document_2_list[term_1, term_2, ... term_n], ..., document_n_list[term_1, term_2, ... term_n]]
    :return:
    """
    df_drug_stems = pd.read_csv(os.path.join(os.getcwd(), "base_data", "USAN_Stems.csv"))
    df_drug_stems["prefix"] = df_drug_stems.apply(lambda x: True if x["Stems"][0] != "-" and x["Stems"][-1] == "-" else False, axis=1)
    df_drug_stems["suffix"] = df_drug_stems.apply(lambda x: True if x["Stems"][0] == "-" and x["Stems"][-1] != "-" else False, axis=1)
    df_drug_stems["infix"] = df_drug_stems.apply(lambda x: True if x["Stems"][0] == "-" and x["Stems"][-1] == "-" else False, axis=1)
    df_drug_stems["just_stem"] = df_drug_stems.apply(lambda x:  x["Stems"].strip("-"), axis=1)
    df_drug_stems["error"] = df_drug_stems.apply(lambda x: False if x["prefix"] or x["suffix"] or x["infix"] else True, axis=1)
    # med_dict = dict(df_med["Synonyms"])
    prefix_set = set(df_drug_stems[df_drug_stems["prefix"]]["just_stem"])
    prefix_len_set = set(len(i) for i in prefix_set)
    suffix_set = set(df_drug_stems[df_drug_stems["suffix"]]["just_stem"])
    suffix_len_set = set(len(i) for i in suffix_set)
    infix_set = set(df_drug_stems[df_drug_stems["infix"]]["just_stem"])
    infix_len_set = set(len(i) for i in infix_set)

    med_terms = [[term for term in doc
                         if any(True for i in suffix_len_set
                                if len(term) > i and term[-i:] in suffix_set)
                         or any(True for i in prefix_len_set
                                if len(term) > i and term[:i] in prefix_set)
                         or any(True for i in infix_len_set
                                for j in range(len(term))
                                if len(term[j+1:-1]) > i and term[j+1:j+1+i]) in infix_set]
                        for doc in document_text_list]

    d = enchant.Dict("en_US")

    med_terms = [[term for term in doc if not d.check(term)] for doc in med_terms]
    if config["NLP"]["filter_medical_terms"]:
        med_terms = filter_medical_terms(document_text_list=med_terms, config=config)
    return med_terms


def filter_medical_terms(document_text_list: list, config: dict):
    ignore_terms = set(config["NLP"]["medicalterms_to_ignore"])
    ignore_term_suffix = config["NLP"]["medicalterms_to_ignore_suffix"]
    document_text_list = [[term for term in doc if not term in ignore_terms] for doc in document_text_list]
    document_text_list = [[term for term in doc
                  if not any(i == term[-len(i):] for i in ignore_term_suffix)]
                 for doc in document_text_list]
    return document_text_list


def extract_abbreviations(document_text_list: list):
    return [[term for term in doc if bool(re.match(r'\w*[A-Z]\w*', term))] for doc in document_text_list]


def extract_disease_terms(document_text_list: list, config: dict):
    """
    :param document_text_list: list -> corpus_list[document_1_list[term_1, term_2, ... term_n], document_2_list[term_1, term_2, ... term_n], ..., document_n_list[term_1, term_2, ... term_n]]
    :return:
    """
    df_disease = pd.read_csv(os.path.join(os.getcwd(), "base_data", "disease_suffix.csv"))
    df_disease["just_stem"] = df_disease.apply(lambda x: x["SUFFIX"].strip("-"), axis=1)

    suffix_set = set(df_disease["just_stem"])
    suffix_len_set = set(len(i) for i in suffix_set)


    disease_terms = [[term for term in doc
                         if any(True for i in suffix_len_set
                                if len(term) > i and term[-i:] in suffix_set)]
                        for doc in document_text_list]
    ignore_terms = config["NLP"]["disease_terms_to_ignore"]
    disease_terms = [[term for term in doc if not term in ignore_terms] for doc in disease_terms]
    # d = enchant.Dict("en_US")
    #
    # disease_terms_non_english = [[term for term in doc if not d.check(term)] for doc in disease_terms]

    return disease_terms



def identify_possible_diseases(document_text_list: list, terms_of_interest: set, config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    target_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords",
                               f'disease_name_counts_top_{config["Query"]["top_x_1_gram_disease_names"]}.pkl')
    if os.path.isfile(target_file):
        return

    ngram_list = [[[" ".join([i for i in ngram]) for ngram in ngrams(sequence=doc, n=n)] for doc in document_text_list] for n in
                  [2, 3, 4, 5]]
    c_dict = count_dict_three_level(ngram_list)
    c_dict = {k: v for k, v in c_dict.items() if v > config["Query"]["minimum_naming"]
              and any(True if term in k else False for term in terms_of_interest)}
    c_dict = {k: v for k, v in c_dict.items() if not "be" in k}
    new_dict = {}
    for k, v in c_dict.items():

        dict_of_duplicates = {key: val for key, val in c_dict.items() if (k in key or key in k) and k != key}
        if not any(v < dup_val for dup_val in dict_of_duplicates.values()):
            new_dict[k] = v

    df_disease_count = pd.DataFrame(data={"disease name": new_dict.keys(), "term count": new_dict.values()})
    write_pickle(path=target_file, obj=df_disease_count, logger=logger)
    

def identify_possible_diseases_wrapper(config: dict, document_text_list: list):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    target_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords",
                               f'disease_name_counts_top_{config["Query"]["top_x_1_gram_disease_names"]}.pkl')
    if os.path.isfile(target_file):
        return

    origin_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", "1", "tfidf_1_gram_disease_filterable.pkl")

    df = read_pickle(path=origin_file, logger=logger)
    df = df.nlargest(20, columns=[config["NLP"]["tfidf_filter_criterion"]])
    df = df.sort_values(by="sum", ascending=False)
    if config["Query"]["top_x_1_gram_disease_names"] == "all":
        terms_of_interest = set(df.index)
    else:
        terms_of_interest = set(df.head(config["Query"]["top_x_1_gram_disease_names"]).index)
    identify_possible_diseases(document_text_list=document_text_list, terms_of_interest=terms_of_interest, config=config)

if __name__ == "__main__":
    pass