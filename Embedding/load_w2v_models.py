import os, logging, datetime


from misc.abb_med_disease import gen_abb_med_disease
from misc.save_files import read_DocTermDict, read_w2v_model


def load_models(config: dict, year_list: list, med: bool, abb: bool, disease: bool):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    target_dir = os.path.join(config["System"]["base_dir"], config["NLP"]["skip_gram_CBOW"])
    if not os.path.isdir(target_dir):
        logger.info(f"No Embeddings found at: {target_dir}")
        return None, None, None

    if year_list is None:
        year_list = list(reversed(range(datetime.date.today().year, config["NLP"]["min_year"] - 1,
                                        -config["NLP"]["year_interval"])))

    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

    if (not abb and not med and not disease):
        model_dir = target_dir
    else:
        model_dir = os.path.join(target_dir, abb_med)

    if os.path.isdir(model_dir):  # use the med abb directory
        model_dict = {year: read_w2v_model(path=os.path.join(model_dir, f"w2v_{year}.kv"), logger=logger)
                 for year in year_list}
    else:
        logger.error(f"Model Target Directory does not exist: {model_dir}")
        return None, None, None
    term_dict = read_DocTermDict(target_dir=os.path.join(config["System"]["base_dir"], f"DocumentTermDictionary",
                                                         f"{abb_med}"),
                                 logger=logger)

    return model_dict, term_dict, year_list


def load_joined_models(config: dict, joined_query: str, year_list: list, med: bool, abb: bool, disease: bool):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    target_dir = os.path.join(config["System"]["base_dir"], config["NLP"]["skip_gram_CBOW"], joined_query)
    if not os.path.isdir(target_dir):
        logger.info(f"No Embeddings found at: {target_dir}")
        return None, None, None

    if year_list is None:
        year_list = list(reversed(range(datetime.date.today().year, 2000,
                                        -config["NLP"]["year_interval"])))

    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

    if (not abb and not med and not disease):
        model_dir = target_dir
    else:
        model_dir = os.path.join(target_dir, abb_med)
    if os.path.isdir(model_dir):  # use the med abb directory
        model_dict = {year: read_w2v_model(path=os.path.join(model_dir, f"w2v_{year}.kv"), logger=logger)
                 for year in year_list}
        model_dict = {k: v for k, v in model_dict.items() if v is not None}
    else:
        logger.error(f"Model Target Directory does not exist: {model_dir}")
        return None, None, None
    term_dict = read_DocTermDict(target_dir=model_dir,
                                 logger=logger)

    return model_dict, term_dict, year_list


def load_current_joined_model(config: dict, joined_query_term: str):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    target_dir = os.path.join(config["System"]["base_dir"], config["NLP"]["skip_gram_CBOW"], joined_query_term)
    if not os.path.isdir(target_dir):
        logger.info(f"No Embeddings found at: {target_dir}")
        return None, None, None

    model = read_w2v_model(path=os.path.join(target_dir, f"w2v_{datetime.date.today().year}.kv"), logger=logger)
    return model
