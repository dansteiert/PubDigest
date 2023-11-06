import pandas as pd
import os, logging
import ast

from misc.save_files import read_pickle, write_pickle, read_pandas_from_csv, write_pandas_to_csv
from misc.counter import count_dict


def dict_of_abbreviations(df: pd.DataFrame, config: dict):
    """
    Go through each publication and check for abbreviations (longer string than 1, 2. letter is a capital letter,
    surrounded by parentheses!)
    Add the words previous to the term (depending on the term length (e.g. API, gets 3 words)) into a list.
    retrieve for each abbreviation the most common occurance found in the dataset, which will be returned as the
    translation of the term.
    Additionally for each term and translation, a count is given, to show the confidence in the translation.
    :param df_paper:y
    :param base_dir:
    :return:
    """
    target_file = os.path.join(config["System"]["base_dir"], "Abbreviations.csv")
    if os.path.isfile(target_file):
        return
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    if df is None:
        df= read_pickle(os.path.join(config["System"]["base_dir"], "dataset.pkl"), logger=logger)

    abb_dict = {term: {"count": 0, "translations": []}
                     for doc in df["abbreviations_clean_text"].tolist()
                     for term in doc}

    # go through the abstract, "tokenize" it, and iterate over each token and check if it is an abbreviation,
    # if so append the X words before the occurance to a list, just looking for abbreviations given in parentheses!
    for doc in df["clean_text"].tolist():
        for idx, term in enumerate(doc):
            if term in abb_dict:
                temp_dict = abb_dict.get(term)
                temp_dict["count"] += 1
                temp_dict["translations"].append(" ".join(doc[idx - len(term): idx]))
                temp_dict["translations"].append(" ".join(doc[idx + 1: idx + 1 + len(term)]))
                abb_dict[term] = temp_dict
    abb_dict_res = {}
    for k, v in abb_dict.items():
        cd = count_dict(l=v["translations"])
        if cd.get("", False):
            del cd[""]
        cd = sorted(cd.items(), key=lambda d: d[1], reverse=True)

        abb_dict_res[k] = {"count": v["count"], **{f"{idx + 1}. translation": i[0] for idx, i in enumerate(cd[:3])},
                                              **{f"{idx+ 1}. translation count": i[1] for idx, i in enumerate(cd[:3])}}

    df_abb = pd.DataFrame(data=abb_dict_res).transpose().reset_index()

    write_pandas_to_csv(path=target_file, logger=logger, df=df_abb)

