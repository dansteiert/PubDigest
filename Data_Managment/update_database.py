import os, datetime, logging

import pandas as pd
from multiprocessing import Pool

from Data_Managment.crawl_data import get_inhouse_id_paths, get_ids, already_in_inhouse_database, full_processed_articles, load_publication_pickle
from misc.save_files import read_pickle, write_pickle, read_pandas_from_csv

from Keywords.special_terms import extract_abbreviations, extract_medical_terms, extract_disease_terms



def data_collection(config: dict, full_query: str):
    """
    Download missing publications, add citations and generate a pd.DataFrame
    :param config: dict of configurations
    :param full_query: the query string, including language and other restrictions
    :return: pd.DataFrame, containing all entries
    """

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info(f"Data Collection - started for: {os.path.split(config['System']['base_dir'])[1]}")

    # <editor-fold desc="Already Known Files for the query">
    query_ids_path = os.path.join(config["System"]["base_dir"], "Query_ids.pkl")
    dataset_path = os.path.join(config["System"]["base_dir"], "dataset.pkl")
    inhouse_ids_path = os.path.join(config["System"]["working_dir"], "data", "inhouse_ids.pkl")
    publication_folder_path = os.path.join(config["System"]["working_dir"], "processed_publications")
    cities_location_path = os.path.join(config["System"]["working_dir"], "base_data", "worldcities.csv")

    query_ids_necessary = update_file_check(path=query_ids_path,
                                            config=config)
    download_missing_files_necessary = update_file_check(path=dataset_path,
                                            config=config)
    # </editor-fold>

    # <editor-fold desc="Select PubMed Ids">
    if not query_ids_necessary or config["Workflow"]["update_dataset"]:
        if config["Query"]["query_ids_given"]:
            try:
                # option if the publication ids are already known and supplied
                df = read_pandas_from_csv(path=config["Query"]["query_ids_file_path"],
                                          sep=config["Query"]["query_ids_file_sep"])
                query_ids = set(df[config["Query"]["query_ids_file_id_col"]])
                del df
            except:
                logger.fatal("Data Collection - Query ID file is not readable")
                exit()
        else:
            query_ids = set(get_ids(full_query=full_query, config=config))
    else:
        logger.info("Data Collection - Read publication IDs from disk")
        df_query = read_pickle(query_ids_path, logger=logger)
        query_ids = set(df_query["query_ids"])

    write_pickle(query_ids_path,
                 pd.DataFrame(data={"query_ids": list(query_ids)}), logger=logger)
    logger.info("Data Collection - Publication IDs wrote to disk")
    # </editor-fold>


    # <editor-fold desc="Select, Download and Process Articles">
    if not download_missing_files_necessary \
            or config["Workflow"]["update_dataset"] \
            or config["Workflow"]["sanitise_missing_publications"]:
        if os.path.isfile(inhouse_ids_path):
            inhouse_ids = read_pickle(inhouse_ids_path, logger=logger)
        else:
            inhouse_ids = get_inhouse_id_paths(publication_folder_path)

        missing_ids = list(already_in_inhouse_database(query_ids=query_ids, inhouse_ids=inhouse_ids))
        logger.info(f"Data Collection - The query yielded: ({len(query_ids) - len(missing_ids)}/{len(query_ids)}) "
                    f"(already in the database/publications found) "
                    f"{f', the missing {len(missing_ids)} publications are currently downloaded and processed' if len(missing_ids) > 0 else ''}")

        df_affiliation = read_pandas_from_csv(path=cities_location_path,
                                              logger=logger, index_col=False,
                                              usecols=["city", "lat", "lng", "country", "iso3"], sep=",")
        df_affiliation["index"] = df_affiliation.index
        full_processed_articles(config=config, pubmed_ids=missing_ids, df_affiliation=df_affiliation,
                                inhouse_ids=inhouse_ids)
    else:
        logger.info("Data Collection - No files to download")
    # </editor-fold>

    # # <editor-fold desc="Compile dataset and retrieve citations">
    if not download_missing_files_necessary \
            or config["Workflow"]["update_citations"] \
            or config["Workflow"]["update_dataset"] \
            or config["Workflow"]["sanitise_missing_publications"]:

        if os.path.isfile(dataset_path) \
                and config["Workflow"]["skip_build_dataset"] \
                and len(missing_ids) > config["Workflow"]["additional_publications_dataset_rebuild_threshold"]:

            # <editor-fold desc="Read Data From Disk">
            df = read_pickle(dataset_path, logger=logger)
            logger.info(f"Data Collection - Dataset read from disk. Datasets contains {df.shape[0]} entries.")
            # </editor-fold>
        else:
            logger.info("Data Collection - Start compiling Dataset")

            # <editor-fold desc="Get PubMed Ids">
            if os.path.isfile(inhouse_ids_path):
                inhouse_ids = read_pickle(inhouse_ids_path, logger=logger)
            else:
                inhouse_ids = get_inhouse_id_paths(publication_folder_path)
            error_ids = already_in_inhouse_database(query_ids=query_ids, inhouse_ids=inhouse_ids)
            logger.info(f"Data Collection - IDs retrieved: {len(query_ids)}. "
                        f"An error occured for: {len(error_ids)} publications"
                        f"{f'. Their IDs are: {error_ids}' if len(error_ids) > 0 else ''}")

            query_ids.difference_update(error_ids)
            query_id_list = list(query_ids)
            # </editor-fold>

            # <editor-fold desc="Compile Dataset">
            logger.info(f"Data Collection - Start compile dataset in batches of {config['System']['dataset_compile_batch_size']}:")
            for batch_no, batches in enumerate(range(0, len(query_ids), config["System"]["dataset_compile_batch_size"])):
                with Pool(processes=config["System"]["processors"]) as pool:
                    write_pickle(os.path.join(config["System"]["base_dir"], f"dataset_batch_{batch_no}.pkl"),
                                 pd.DataFrame(pool.imap_unordered(load_publication_pickle,
                                                                  [inhouse_ids.get(i, f"No path for this ID found: {i}")
                                                                   for i in query_id_list[batches: batches + config["System"]["dataset_compile_batch_size"]]
                                                                   ], chunksize=5000)), logger=logger)

                print("#", end="")
            logger.info(f"Data Collection - Dataset batches compiled - compile large dataset")

            df = pd.concat([read_pickle(os.path.join(config["System"]["base_dir"], i), logger=logger)
                            for i in os.listdir(config["System"]["base_dir"])
                            if "dataset_batch" in i],
                           ignore_index=True)
            logger.info("Data Collection - Dataset fully compiled")


            try:
                df.set_index(keys="identifier", inplace=True)
            except Exception as e:
                logger.fatal(f"Data Collection - {e}, {df.columns.tolist()}")
                exit()
            # </editor-fold>

            # <editor-fold desc="Remove Data with missing entries">

            df = df[~df.isna().all(axis=1)]
            df = df[~df["PMID"].isna()]
            df = df.astype({"PMID": int, "publication year": int})

            # </editor-fold>

            # <editor-fold desc="Define Special Terms">
            df["medical_terms_clean_text"] = extract_medical_terms(
                document_text_list=df["clean_text"].tolist(), config=config)

            logger.info("Filtered Medical Terms for enzyms")
            df["abbreviations_clean_text"] = extract_abbreviations(
                document_text_list=df["clean_text"].tolist())
            df["abb_med_terms_clean_text"] = df["abbreviations_clean_text"] + df["medical_terms_clean_text"]
            df["disease_terms_clean_text"] = extract_disease_terms(document_text_list=df["clean_text"].tolist(),
                                                                   config=config)
            # </editor-fold>

            write_pickle(os.path.join(config["System"]["base_dir"], "dataset.pkl"), df, logger=logger)
            logger.info(f"Data Collection - Dataset compiled and citations retrieved. Datasets contains {df.shape[0]} entries.")
    else:
        df = read_pickle(dataset_path, logger=logger)
        logger.info(f"Data Collection - Dataset read from disk. Datasets contains {df.shape[0]} entries.")
    # </editor-fold>
    return df



def update_file_check(path: str, config: dict):
    isfile = os.path.isfile(path)
    if not isfile:
        return False
    time_since_change = (datetime.datetime.now() - datetime.datetime.fromtimestamp(
        max(os.path.getctime(path),
            os.path.getmtime(path)))).days
    return time_since_change < config["Query"]["max_days_since_last_crawl"]



if __name__ == "__main__":
    pass
    