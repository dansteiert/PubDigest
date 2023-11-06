import logging, os, datetime
import pandas as pd

from Embedding.retrospective_validation import single_model_term_similarity, is_in_dataset
from misc.save_files import write_pickle, read_DocTermDict, read_pickle
from Embedding.load_w2v_models import load_current_joined_model

def specific_term_validation(config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])



    for query_term in config["Query"]["alternate_query_terms"]:

        target_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Embedding",
                                   f"specific_term_validation_{config['NLP']['skip_gram_CBOW']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}.pkl")
        if os.path.isfile(target_file):
            continue

        term_dict = read_DocTermDict(
            target_dir=os.path.join(config["System"]["base_dir"], config["NLP"]["skip_gram_CBOW"],
                                    query_term),
            logger=logger)
        model = load_current_joined_model(config=config, joined_query_term=query_term)
        if model is None:
            logger.error(f"No model found for: {config['Query']['base_query_term']} & {query_term} for year: {datetime.date.today().year}")

        result_list = []
        for i in config["Query"]["terms_for_embedding_evaluation"]:
            for term in i:
                result_list.extend(single_model_term_similarity(config=config, model=model, term_dict=term_dict,
                                                                base_term=config["Query"]["prediction_term"],
                                                                query_term=term, logger=logger))
        df = pd.DataFrame(data=result_list)
        if df.empty:
            continue

        write_pickle(path=target_file, obj=df, logger=logger)
        is_in_dataset(config=config, dataset_a=config["Query"]["base_query_term"], dataset_b=query_term,
                      term_set=set(df["query_term"]), specifier="specific_term_validation")

    return
