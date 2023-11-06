import os, pickle, sys, datetime, logging
import pandas as pd


sys.path.append(os.path.join(os.getcwd()))
from Data_Managment.publication_sanitisation import sanitise_publication

# APIs
from pymed import PubMed
from misc.Icite_API import get_citation_single, get_citations
from multiprocessing import Pool
from misc.save_files import write_pickle


def full_processed_articles(config: dict, pubmed_ids: list,
                            df_affiliation: pd.DataFrame, inhouse_ids: dict):

    # <editor-fold desc="Setup">
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    if len(pubmed_ids) == 0:
        logger.info("PubMed Article Download - Skipped - Nothing new to Download")
        return
    pubmed = PubMed(tool=config["PMC"]["tool_name"], email=config["PMC"]["email"])

    num_of_ids = len(pubmed_ids)
    batch_size = config["PMC"]["pymed"]["batch_size"]

    logger.info("PubMed Article Download - Batch Download Started:")
    # </editor-fold>

    # TODO: set timelimit to the three steps get publications, get citations and processing -> sometimes it gets stuck
    # with Pool(processes=config["System"]["processors"]) as pool:
    #     pool.starmap(pubmed.get_publications_from_ids, [(pubmed_ids[index: min(index + batch_size, num_of_ids)], config["PMC"]["pymed"]["timeout"])
    #                                         for index in range(0, num_of_ids, batch_size)])
    for index in range(0, num_of_ids, batch_size):
        # <editor-fold desc="Download Publications">
        counter = 0
        while True:
            try:
                crawled = pubmed.get_publications_from_ids(
                    article_ids=pubmed_ids[index: min(index + batch_size, num_of_ids)],
                    # max_results=config["PMC"]["pymed"]["max_results"],
                    timeout=config["PMC"]["pymed"]["timeout"])
                crawled_articles = list(crawled)
                break
            except:
                counter +=1
                if counter >= 5:
                    logger.error(f"PubMed Article Download - restart the pipeline to retrive the missing datapoints\n"
                                 f"\t\tFrom {num_of_ids}, {index} have been properly downloaded")
                    break


        crawled_citations = get_citations(pmids=pubmed_ids[index: min(index + batch_size, num_of_ids)],
                                          fields_to_return=config["PMC"]["iCite"]["fields_to_return"],
                                          chunk_size=config["PMC"]["iCite"]["chunck_size"])
        # </editor-fold>

        # <editor-fold desc="process Publications">
        if len(crawled_articles) > 0:
            with Pool(processes=config["System"]["processors"]) as pool:
                processed_ids = pool.starmap(sanitise_publication, [(config, article_obj, iCite_dict,
                                                                     df_affiliation, pubmed_ids[index: min(index + batch_size, num_of_ids)])
                                                                    for (article_obj, iCite_dict) in
                                                                    zip(crawled_articles, crawled_citations)])
        else:
            continue
        # </editor-fold>

        # <editor-fold desc="Save Publication Data">
        inhouse_ids = {**inhouse_ids, **{k: v for d in processed_ids for k, v in d.items()}}
        write_pickle(path=os.path.join(config["System"]["working_dir"], "data", "inhouse_ids.pkl"), obj=inhouse_ids, logger=logger)
        print("#", end="")
        # </editor-fold>


def get_ids(full_query: str, config: dict):
    pubmed = PubMed(tool=config["PMC"]["tool_name"], email=config["PMC"]["email"])
    crawled = pubmed.query_publication_ids(query=full_query, max_results=config["PMC"]["pymed"]["max_results"],
                                           timeout=config["PMC"]["pymed"]["timeout"])
    return crawled


def get_inhouse_id_paths(path):
    if not os.path.isdir(path):
        return {}
    path_dict = {}
    for i in os.listdir(path):
        new_path = os.path.join(path, i)
        if os.path.isfile(new_path):
            path_dict[i[:-4]] = new_path
        elif os.path.isdir(new_path):
            path_dict = {**path_dict, **get_inhouse_id_paths(new_path)}
    return path_dict


def already_in_inhouse_database(query_ids, inhouse_ids):
    return set(query_ids).difference(set(inhouse_ids.keys()))


def load_publication_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return {}


def retrieve_citation_count_for_dataframe(df: pd.DataFrame, config: dict):
    if not os.path.isfile(os.path.join(config["System"]["base_dir"], "iCite.pkl")) or \
            config["Workflow"]["update_citations"] or \
            config["Workflow"]["update_citations_override"]:
        iCite_results = get_citations(pmids=df.index.tolist(),
                                      fields_to_return=config["PMC"]["iCite"]["fields_to_return"],
                                      chunk_size=config["PMC"]["iCite"]["chunck_size"])
        df_citations = pd.DataFrame(data=iCite_results)
        df_citations['pmid'] = df_citations['pmid'].astype("string")
        df_citations.to_pickle(os.path.join(config["System"]["base_dir"], "iCite.pkl"))
    else:
        df_citations = pd.read_pickle(os.path.join(config["System"]["base_dir"], "iCite.pkl"))

    df = df.join(other=df_citations.set_index("pmid"), how="outer", rsuffix="_citations")
    return df


def retrieve_citation_count_for_single_entry(article_dict: dict, config: dict):
    article_dict = {**article_dict,
                    **get_citation_single(pmid=article_dict["identifier"],
                                          fields_to_return=config["PMC"]["iCite"]["fields_to_return"])}
    article_dict["citation_update_year"] = datetime.datetime.now().year
    article_dict["citation_update_month"] = datetime.datetime.now().month
    article_dict["citation_update_day"] = datetime.datetime.now().day
    return article_dict
