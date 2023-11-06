import os, logging

import nltk
try:
    from nltk.tokenize import TweetTokenizer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    from nltk.util import ngrams
except:
    nltk.download()

from gensim.models import TfidfModel
from gensim import corpora

import pandas as pd

from misc.save_files import write_pickle
from misc.abb_med_disease import gen_abb_med_disease

def tfidf_wrapper(df: pd.DataFrame, config: dict):
    for n_gram in config["NLP"]["n_gram_list"]:
        tfidf(config=config, lemma_list=df["clean_text"].tolist(), year_list=df["publication year"].tolist(),
              abb=False, med=False, n_gram=n_gram, disease=False)
        tfidf(config=config, lemma_list=df["medical_terms_clean_text"].tolist(),
              year_list=df["publication year"].tolist(),
              abb=False, med=True, n_gram=n_gram, disease=False)
        tfidf(config=config, lemma_list=df["abbreviations_clean_text"].tolist(),
              year_list=df["publication year"].tolist(),
              abb=True, med=False, n_gram=n_gram, disease=False)
        tfidf(config=config, lemma_list=[[*abb, *med] for abb, med in zip(df["abbreviations_clean_text"].tolist(),
                                                                          df["medical_terms_clean_text"].tolist())],
              year_list=df["publication year"].tolist(),
              abb=True, med=True, n_gram=n_gram, disease=False)
        tfidf(config=config, lemma_list=df["disease_terms_clean_text"].tolist(),
              year_list=df["publication year"].tolist(),
              abb=False, med=False, n_gram=n_gram, disease=True)


def tfidf(config: dict, lemma_list: list, year_list: list, abb: bool, med: bool, n_gram: int, disease: bool):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    target_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords",
                               f"{n_gram}", f"tfidf_{n_gram}_gram{abb_med}.pkl")
    if os.path.isfile(target_file):
        return

    if n_gram > 1:
        ngram_list = [[" ".join([i for i in ngram]) for ngram in ngrams(sequence=doc, n=n_gram)] for doc in lemma_list]
    else:
        ngram_list = lemma_list

    # <editor-fold desc="generate dictionary and remove those entries below threshold">
    dictionary = corpora.Dictionary(ngram_list)
    dictionary.filter_extremes(no_below=2)
    # </editor-fold>

    # <editor-fold desc="cut short if no elements remain">
    if len(dictionary.token2id) == 0:
        return None
    # </editor-fold>

    # <editor-fold desc="generate the tfidf model">
    BoW_corpus = [dictionary.doc2bow(doc) for doc in ngram_list]
    tfidf_model = TfidfModel(corpus=BoW_corpus, dictionary=dictionary, smartirs='ntc')
    # </editor-fold>

    # <editor-fold desc="generate a csv for later, by splitting it into years">
    # have the idfs separately, get the frequency by year (Needs division by publication count per timeinterval!!)
    idf_dict = tfidf_model.idfs  # id: idf
    idf_dict = {dictionary.id2token.get(k, ""): v for k, v in idf_dict.items()} # term: idf
    year_dict = {**{"idf": idf_dict},
                 **{i: {} for i in range(int(min(year_list)), int(max(year_list)))}}  # year: {term: tf (empty except for idf)}
    for doc, year in zip(tfidf_model[BoW_corpus], year_list): # fill year_dict with {term: tf} dictionaries
        temp_dict = year_dict.get(int(float(year)), {})  # {term: tf}
        doc_dict = {dictionary.get(idx, ""): freq for idx, freq in doc}  # doc = idx, tf-idf -> doc_dict = {term: tf-idf}

        year_dict[int(float(year))] = {k: temp_dict.get(k, 0) + (doc_dict.get(k, 0) / idf_dict.get(k, 0)) # add tf of known terms to tf-idf/idf=tf
                                       for k in set([*temp_dict.keys(), *doc_dict.keys()])}# iterate over all terms for this year
    # </editor-fold>

    # <editor-fold desc="Sort years ascending and sort the values ascending starting by the earliest year">
    df = pd.DataFrame(data=year_dict)
    df = df.fillna(0)
    df = df.reindex(columns=[*sorted([i for i in df.columns if i != "idf"]), "idf"])
    df = df.sort_values(by=sorted([i for i in df.columns if i != "idf"]))
    # </editor-fold>

    # <editor-fold desc="save results as csv and filter">
    if df.shape[0] == 0:
        return None

    write_pickle(path=target_file,
                        obj=df, logger=logger)
    # </editor-fold>
