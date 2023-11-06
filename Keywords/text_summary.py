import pandas as pd
import os, logging

from misc.abb_med_disease import gen_abb_med_disease
from misc.save_files import read_pickle, read_DocTermDict


def gen_summary_text_wrapper(config: dict):
    disease = False
    for med in [True, False]:
        for abb in [True, False]:
            abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
            for n_gram in config["NLP"]["n_gram_list"]:
                summary_file(config=config, n_gram=n_gram, abb_med=abb_med)
    disease = True
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    for n_gram in config["NLP"]["n_gram_list"]:
        summary_file(config=config, n_gram=n_gram, abb_med=abb_med)

def summary_file(config: dict, n_gram: int, abb_med: str, top_description_terms: int = 5):
    result_dict = top_keywords(config=config, n_gram=n_gram, abb_med=abb_med)
    file_dir = os.path.join(config["System"]["base_dir"], "Text Summary")
    os.makedirs(file_dir, exist_ok=True)

    with open(os.path.join(file_dir, f"summary{abb_med}.txt"), "w", encoding="utf-8") as f:
        f.write(f"PubMed Query: {config['Query']['base_query_term']}\n")
        f.write(f"Found Publications: {result_dict.get('publication_count', None)}\n")
        f.write(f"Most published authors are:\n")
        for k, v in sorted(result_dict.get("authors", {}).items(),
                           key=lambda d: d[1].get("publication_count", 0), reverse=True):
            f.write(
                f"\t{v.get('reference name', None)}\twith {v.get('publication_count', None)} publications, "
                f"\twhere {v.get('first_author', None)}, "
                f"{v.get('last_author', None)} "
                f"are published as first and last author respectively; The publications received a total of "
                f"{v.get('citation count')} citations\n")
        f.write(f"Highest cited publications are:\n")
        for k, v in sorted(result_dict.get("publication_citations", {}).items(),
                           key=lambda d: d[1].get("iCite_citation_count", 0), reverse=True):
            f.write(
                f"\tPubMed ID: {v.get('pmid')[0]}; Citation count: {v.get('iCite_citation_count', None)}; "
                f"\tTitle: {v.get('title', None)} \n")
        f.write(f"The most important terms in this research area are:\n")
        for k, v in sorted(result_dict.get("titles_num", {}).items(), key=lambda d: d[1], reverse=True):
            f.write(f"\tMentions: {v}\tTerm: {k}\n")
                     #f"\t{', '.join([i[0] for i in v[:top_description_terms]])}\n") #  TODO: derive closely related terms (word2vec?)


def top_keywords(config: dict, n_gram: int, abb_med: str):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    df_tfidf = read_pickle(path=os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                                       f"tfidf_{n_gram}_gram{abb_med}_filterable.pkl"), logger=logger)
    if df_tfidf is None:
        return {}
    df_publication = read_pickle(path=os.path.join(config["System"]["base_dir"], "dataset.pkl"), logger=logger)
    df_authors = read_pickle(path=os.path.join(config["System"]["base_dir"], "authors.pkl"), logger=logger)


    term_dict = read_DocTermDict(target_dir=os.path.join(config["System"]["base_dir"], f"DocumentTermDictionary",
                                                         f"{abb_med}"), logger=logger)

    df_tfidf.sort_values(by="avg", ascending=False, inplace=True)

    most_important_terms = list(df_tfidf.index.values)[:config["NLP"]["summary_top_x_terms"]]



    titles_num = {i: len(term_dict.term2key_doc.get(i, None)[1])
              for i in most_important_terms} # search the titles for the most important terms


    
    return {"titles_num": titles_num, "publication_count": df_publication.shape[0],
            "publication_citations": get_most_cited_publications(df=df_publication, top_cited=config["NLP"]["summary_top_x_publications"]),
            "authors": get_most_published_authors(df=df_authors, top_published=config["NLP"]["summary_top_x_authors"])}


def get_most_cited_publications(df: pd.DataFrame, top_cited: int = 10):
    df.sort_values(by="iCite_citation_count", ascending=False, inplace=True)
    return df.head(top_cited).to_dict(orient="index")


def get_most_published_authors(df: pd.DataFrame, top_published: int = 10):
    df.sort_values(by="publication_count", ascending=False, inplace=True)
    return df.head(top_published).to_dict(orient="index")



if __name__ == "__main__":
    pass

