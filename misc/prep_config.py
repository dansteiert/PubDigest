import os
import yaml
from misc.setup_logger import setup_logger

def load_configs(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        with open(v) as f:
            config[k] = yaml.safe_load(f)

    config = prep_config(config=config)
    return config


def prep_config(config: dict):
    """
    Setup important paths, number of threads, query parameters and logger
    :param config: dictionary of configurations
    :return: config
    """
    os.environ["NUMEXPR_NUM_THREADS"] = str(config["System"]["processors"])


    # <editor-fold desc="Make sure query terms are lower case">
    config["Query"]["alternate_query_terms"] = [i.lower() for i in config["Query"]["alternate_query_terms"]]
    config["Query"]["base_query_term"] = config["Query"]["base_query_term"].lower()
    # </editor-fold>
    
    # <editor-fold desc="Set directory paths">
    if config["System"]["working_dir"] is None:
        config["System"]["working_dir"] = os.getcwd()


    # generate Folders
    config["System"]["base_dir"] = os.path.join(config["System"]["working_dir"], "data", config["Query"]["base_query_term"])
    os.makedirs(config["System"]["base_dir"], exist_ok=True)

    [os.makedirs(os.path.join(config["System"]["working_dir"], "data", i), exist_ok=True)
     for i in config["Query"]["alternate_query_terms"]]


    # Generate Query Strings
    config["Query"]["base_query_parameters"]["full_query"] = " AND ".join([f"{v}[{k}]"
                                                                           for k, v in {**{"": config["Query"]["base_query_term"]},
                                                                                        **config["Query"]["base_query_parameters"]}.items()
                                                                           if v is not None])

    config["Query"]["alternate_query_parameters"]["full_query"] = [
        " AND ".join([f"{v}[{k}]"
                      for k, v in {**{"": i}, **config["Query"]["alternate_query_parameters"]}.items()
                      if v is not None])
        for index, i in enumerate(config["Query"]["alternate_query_terms"])
    ]


    # Overwrite logging file
    config["System"]["logging"]["filename"] = os.path.join(config["System"]["working_dir"], config["System"]["logging"]["filename"])
    if os.path.isfile(config["System"]["logging"]["filename"]):
        os.remove(config["System"]["logging"]["filename"])

    config = setup_logger(config)

    ## Setup Embedding Prediction
    config["NLP"]["min_year"] = min(config["Query"]["prediction_year"],config["NLP"]["min_year"])
    config["Query"]["terms_to_keep_in_embedding"] = set([*config["Query"]["terms_to_keep_in_embedding"],
                                                         config["Query"]["prediction_term"]])

    return config


