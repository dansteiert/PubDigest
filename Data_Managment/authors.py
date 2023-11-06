import os, sys, logging, re
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from misc.save_files import read_pickle, read_pandas_from_csv, write_pickle
from misc.counter import get_most_occuring_element

from Data_Managment.citation_analysis import citation_analysis

def generate_author_table(config: dict, df_publications: pd.DataFrame):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Author Table generation and Dataset Table update -  Start")
    df_author = read_pickle(os.path.join(config["System"]["base_dir"], "authors.pkl"), logger=logger)
    if df_author is not None:
        logger.info("Author Table -  Read from Disk")
        return df_author
    if df_publications is None:
        logger.error("Publications Table not given")
        exit(10)

    # Collect already generated Data
    df_authors_temp = pd.concat([pd.DataFrame(i) for i in df_publications["authors"].tolist()], ignore_index=True)
    df_authors_temp.fillna(np.nan, inplace=True)

    # Combine Authors
    groupby = df_authors_temp.groupby(by=["lastname", "firstname"])

    df_authors = groupby.agg(lambda srs: np.sort([i[0] for i in Counter(list(srs)).most_common()])[0])

    # Special Cases
    df_authors["first_author"] = groupby["first_author"].agg(lambda srs: Counter(list(srs)).get(True, 0))
    df_authors["last_author"] = groupby["last_author"].agg(lambda srs: Counter(list(srs)).get(True, 0))
    df_authors["second_last_author"] = groupby["second_last_author"].agg(lambda srs: Counter(list(srs)).get(True, 0))
    df_authors["publication_count"] = groupby["pmid"].count()
    df_authors["pmid"] = groupby["pmid"].agg(list)

    # Add First and Last Author Ship PMIDs
    df_authors_temp.dropna(subset=["firstname", "lastname", "first_author", "last_author"], inplace=True)
    df_authors_temp_first = df_authors_temp[df_authors_temp["first_author"]]
    groupby_first = df_authors_temp_first.groupby(by=["lastname", "firstname"])
    series_first_author_pmid = groupby_first["pmid"].agg(list).rename("pmid_first_author")

    df_authors_temp_last = df_authors_temp[df_authors_temp["last_author"]]
    groupby_last = df_authors_temp_last.groupby(by=["lastname", "firstname"])
    series_last_author_pmid = groupby_last["pmid"].agg(list).rename("pmid_last_author")

    df_authors = df_authors.join(series_first_author_pmid)
    df_authors = df_authors.join(series_last_author_pmid)


    df_authors = df_authors.reset_index()
    df_authors["author_id"] = df_authors.index

    df_authors = citation_analysis(df_publications=df_publications, df_authors=df_authors, config=config)

    write_pickle(path=os.path.join(config["System"]["base_dir"], "authors.pkl"), obj=df_authors, logger=logger)

    logger.info("Author Table generation -  Finished")
    return df_authors





if __name__ == "__main__":
    pass
