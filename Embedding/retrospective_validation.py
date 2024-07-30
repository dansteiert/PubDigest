import logging, os, datetime
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from Embedding.prediction import prediction

from misc.abb_med_disease import gen_abb_med_disease
from Data_Managment.build_DocTermDict import DocumentTermDictionary, update_doc_term_dict_instance, select_terms_to_keep_doc_term_dict_instance, get_earliest_meantion_of_term
from misc.counter import count_dict
from misc.save_files import write_pickle, read_DocTermDict
from Embedding.load_w2v_models import load_joined_models


def retrospective_validation_wrapper(config: dict):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])



    for query_term in config["Query"]["alternate_query_terms"]:
        base_term_dict = read_DocTermDict(
            target_dir=os.path.join(config["System"]["base_dir"], config["NLP"]["skip_gram_CBOW"],
                                    query_term),
            logger=logger)
        base_term_dict = select_terms_to_keep_doc_term_dict_instance(term_dict=base_term_dict, terms_to_keep=None, # generated from config!
                                                                     config=config)
        term_set = set()
        disease = False
        for med in [True, False]:
            for abb in [True, False]:
                model_dict, term_dict, year_list = load_joined_models(config=config, joined_query=query_term, year_list=None,
                                                                      med=med, abb=abb, disease=disease)
                if model_dict is None:
                    logger.warning(f"No Models found for: Base Term: {config['Query']['base_query_term']}"
                                   f" and Query Term {config['Query']['alternate_query_term']} - "
                                   f"Abb: {abb}, Med: {med}, Disease: {disease}")
                    continue
                term_dict = update_doc_term_dict_instance(term_dict=term_dict, term_dict_addition=base_term_dict, config=config)
                term_set.update(retrospective_validation(model_dict=model_dict, config=config, term_dict=term_dict,
                                         base_term=config["Query"]["prediction_term"], query_term=query_term,
                                         prediction_year=config["Query"]["prediction_year"], abb=abb, med=med, disease=disease))
                # generate_tsne(config=config, model_dict=model_dict, prediction_year=config["Query"]["prediction_year"],
                #               abb=abb, med=med, disease=disease, dataset_a=config["Query"]["base_query_term"],
                #               dataset_b=query_term, logger=logger
                #
                #               )
                # generate_LLE(config=config, model_dict=model_dict, prediction_year=config["Query"]["prediction_year"],
                #               abb=abb, med=med, disease=disease, dataset_a=config["Query"]["base_query_term"],
                #               dataset_b=query_term, logger=logger
                #
                #               )
        disease = True
        model_dict, term_dict, year_list = load_joined_models(config=config, joined_query=query_term, year_list=None,
                                                              med=med, abb=abb, disease=disease)
        if model_dict is None:
            logger.warning(f"No Models found for: Base Term: {config['Query']['base_query_term']}"
                           f" and Query Term {config['Query']['alternate_query_term']} - "
                           f"Abb: {abb}, Med: {med}, Disease: {disease}")
            continue
        term_set.update(retrospective_validation(model_dict=model_dict, config=config, term_dict=term_dict,
                                 base_term=config["Query"]["prediction_term"], query_term=query_term,
                                 prediction_year=config["Query"]["prediction_year"], abb=abb, med=med, disease=disease))

        is_in_dataset(config=config, dataset_a=config["Query"]["base_query_term"], dataset_b=query_term,
                      term_set=term_set)

        # generate_tsne(config=config, model_dict=model_dict, prediction_year=config["Query"]["prediction_year"],
        #               abb=abb, med=med, disease=disease, dataset_a=config["Query"]["base_query_term"], dataset_b=query_term,
        #               logger=logger)
        # generate_LLE(config=config, model_dict=model_dict, prediction_year=config["Query"]["prediction_year"],
        #               abb=abb, med=med, disease=disease, dataset_a=config["Query"]["base_query_term"], dataset_b=query_term,
        #               logger=logger)

def retrospective_validation(model_dict: dict, term_dict: DocumentTermDictionary, config: dict,
                             base_term: str, query_term: str,
                             prediction_year: int, abb: bool, med: bool, disease: bool):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    target_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Embedding", abb_med,
                                   f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{prediction_year}_{base_term}_{query_term}.pkl")
    if term_dict.term2key_doc.get(base_term, None) is None:
        logger.info(f"Base Term {base_term} is not present in this dataset - Abb: {abb}; Med {med}; Disease {disease}")
        return set()


    query_terms = prediction(model=model_dict.get(prediction_year, None), base_term=base_term,
                             top_prediction_terms=config["Query"]["similar_terms_to_query"])
    if query_terms is None:
        return set()
    result_list = []
    for term, similarity in query_terms:
        result_list.extend(
            query_term_similarity(config=config, model_dict=model_dict, term_dict=term_dict, base_term=base_term,
                                  query_term=term, logger=logger))
    df = pd.DataFrame(data=result_list)
    if df.empty:
        return set()

    write_pickle(path=target_file, obj=df, logger=logger)
    return set(df["query_term"])



def query_term_similarity(config: dict, model_dict: dict, term_dict: DocumentTermDictionary, base_term: str,
                          query_term: str, logger: logging.Logger):

    # get occurances of base and query term, their documents and the years the documents were published
    query_kd = term_dict.term2key_doc.get(query_term, None)
    base_kd = term_dict.term2key_doc.get(base_term, None)

    if base_kd is None:
        logger.info(f"Base Term: {base_term} is not present in the dataset")
        return []
    if query_kd is None:
        logger.info(f"query_term: {query_term} is not present in the dataset")
        return []

    # query_kd and base_kd = (keyword, list(document ids))
    co_occurrence = get_co_occurances(query_set=set(query_kd[1]), sim_set=set(base_kd[1]), term_dict=term_dict)
    if len(co_occurrence) > 0:
        co_occurrence_dict = count_dict([year for doc, year in co_occurrence])
    else:
        co_occurrence_dict = {}

    result_list = []
    for years, model in model_dict.items():
        # catch when  a term is not present in a model!
        try:
            co_occurrence_count = co_occurrence_dict.get(int(years), 0)

            result_list.append({"year": years, "base_term": base_term, "query_term": query_term,
                                "similarity": model.similarity(base_term, query_term),
                                "co_occurrence": co_occurrence_count})
        except:
            continue
    years_before_prediction = [i for i in co_occurrence_dict.keys() if i <= int(config['Query']['prediction_year'])]
    years_after_prediction = [i for i in co_occurrence_dict.keys() if i >= int(config['Query']['prediction_year'])]

    co_occurrence_count = sum([co_occurrence_dict.get(int(i), 0) for i in years_before_prediction])
    result_list.append({"year": f"before_prediction", "base_term": base_term,
                        "query_term": query_term,
                        "similarity": None,
                        "co_occurrence": co_occurrence_count})
    co_occurrence_count = sum([co_occurrence_dict.get(int(i), 0) for i in years_after_prediction])
    result_list.append({"year": f"after_prediction", "base_term": base_term,
                        "query_term": query_term,
                        "similarity": None,
                        "co_occurrence": co_occurrence_count})
    return result_list
def single_model_term_similarity(config: dict, model, term_dict: DocumentTermDictionary, base_term: str,
                          query_term: str, logger: logging.Logger):

    # get occurances of base and query term, their documents and the years the documents were published
    query_kd = term_dict.term2key_doc.get(query_term, None)
    base_kd = term_dict.term2key_doc.get(base_term, None)

    if base_kd is None:
        logger.info(f"Base Term: {base_term} is not present in the dataset")
        return []
    if query_kd is None:
        logger.info(f"query_term: {query_term} is not present in the dataset")
        return []

    # query_kd and base_kd = (keyword, list(document ids))
    co_occurrence = get_co_occurances(query_set=set(query_kd[1]), sim_set=set(base_kd[1]), term_dict=term_dict)
    if len(co_occurrence) > 0:
        co_occurrence_dict = count_dict([year for doc, year in co_occurrence])
    else:
        co_occurrence_dict = {}

    result_list = []
        # catch when  a term is not present in a model!
    year = datetime.date.today().year
    try:
        co_occurrence_count = co_occurrence_dict.get(int(year), 0)

        result_list.append({"year": year, "base_term": base_term, "query_term": query_term,
                            "similarity": model.similarity(base_term, query_term),
                            "co_occurrence": co_occurrence_count})
    except:
        pass
    years_before_prediction = [i for i in co_occurrence_dict.keys() if i <= int(config['Query']['prediction_year'])]

    co_occurrence_count = sum([co_occurrence_dict.get(int(i), 0) for i in years_before_prediction])
    result_list.append({"year": f"before_prediction", "base_term": base_term,
                        "query_term": query_term,
                        "similarity": None,
                        "co_occurrence": co_occurrence_count})
    return result_list


def get_co_occurances(query_set: set, sim_set: set, term_dict: DocumentTermDictionary):
    co_occurrence_set = sim_set.intersection(query_set)
    co_occurrence_list = sorted([(doc, term_dict.doc2year.get(doc, None)) for doc in co_occurrence_set], key=lambda x: x[1])
    return co_occurrence_list


def is_in_dataset(config: dict, dataset_a: str, dataset_b: str,
                  term_set: set, specifier: str = ""):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    target_file = os.path.join(config["System"]["working_dir"], "data", dataset_a, "Data_for_Plotting", "Embedding",
                               dataset_b, f"Prediction_Term_Overlap{'_' + specifier if specifier != '' else ''}.pkl")


    dataset_b_pos = [idx for idx, term in enumerate(config["Query"]["alternate_query_terms"]) if term == dataset_b][0]
    abb_conversion_dict = {dataset_a: config["Query"]["base_query_abb"],
                           dataset_b: config["Query"]["alternate_query_abb"][dataset_b_pos]}

    dataset_a_dict = read_DocTermDict(target_dir=os.path.join(config["System"]["working_dir"], "data", dataset_a,
                                                              f"DocumentTermDictionary"), logger=logger)
    dataset_b_dict = read_DocTermDict(target_dir=os.path.join(config["System"]["working_dir"],"data", dataset_b,
                                                              f"DocumentTermDictionary"), logger=logger)
    if dataset_a_dict is None:
        logger.warning(f"Error in Path: {os.path.join(config['System']['working_dir'], 'data',dataset_a, 'DocumentTermDictionary')}")
    if dataset_a_dict is None:
        logger.warning(f"Error in Path: {os.path.join(config['System']['working_dir'], 'data',dataset_b, 'DocumentTermDictionary')}")

    term_list_dict = [
        {"Term": i, abb_conversion_dict.get(dataset_a, dataset_a): i in dataset_a_dict.term2key_doc,
         abb_conversion_dict.get(dataset_b, dataset_b): i in dataset_b_dict.term2key_doc,
         }
        for i in term_set]
    df = pd.DataFrame(data=term_list_dict).set_index("Term")[[abb_conversion_dict.get(dataset_b, dataset_b), abb_conversion_dict.get(dataset_a, dataset_a)]]

    write_pickle(target_file, obj=df, logger=logger)


def generate_tsne(config: dict, model_dict: dict, dataset_a: str, dataset_b: str,
                  logger: logging.Logger, prediction_year: int, abb: bool, med: bool, disease: bool):
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

    target_file = os.path.join(config["System"]["working_dir"], "data", dataset_a, "Data_for_Plotting", "Embedding",
                               dataset_b, f"Embedding_tsne{abb_med}.pkl")
    if os.path.isfile(target_file):
        return

    model = model_dict[prediction_year]
    vocab = list(model.wv.index2entity)
    X = model.wv[vocab]

    tsne = TSNE(n_components=2, n_jobs=config["System"]["processors"])
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=["x", "y"])
    write_pickle(path=target_file, obj=df, logger=logger)


def generate_LLE(config: dict, model_dict: dict, dataset_a: str, dataset_b: str,
                  logger: logging.Logger, prediction_year: int, abb: bool, med: bool, disease: bool):
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

    target_file = os.path.join(config["System"]["working_dir"], "data", dataset_a, "Data_for_Plotting", "Embedding",
                               dataset_b, f"Embedding_LLE{abb_med}.pkl")
    if os.path.isfile(target_file):
        return

    model = model_dict[prediction_year]
    vocab = list(model.wv.index2entity)
    X = model.wv[vocab]

    lle = LocallyLinearEmbedding(n_components=2, n_jobs=config["System"]["processors"])
    X_lle = lle.fit_transform(X)
    df = pd.DataFrame(X_lle, index=vocab, columns=["x", "y"])
    write_pickle(path=target_file, obj=df, logger=logger)


