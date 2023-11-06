import os, sys, logging
sys.path.append(os.getcwd())

# Setup
from misc.prep_config import load_configs

# Data Acqusition
from Data_Managment.update_database import data_collection
from Data_Managment.publication_metadata import publication_metadata
from Data_Managment.authors import generate_author_table
from Data_Managment.build_DocTermDict import build_doc_term_dict


# Embedding
from Embedding.train_embedding import train_w2v_model, embedding_wrapper_multi_dataset
from Embedding.retrospective_validation import retrospective_validation_wrapper
from Embedding.specific_term_validation import specific_term_validation

# TF-IDF and Keywords
from Keywords.TF_IDF import tfidf_wrapper
from Keywords.special_terms import identify_possible_diseases_wrapper
from Keywords.Time_intervals import generate_keyword_plotting_files

# Text Summary
from Keywords.text_summary import gen_summary_text_wrapper

# DIct of Abbreviations
from Keywords.Dict_of_abbreviations import dict_of_abbreviations

# Visualisations
from Visualization.Visualisations_wrapper import generate_visualisations


def main(config_path):

    config = load_configs(config_path=config_path)
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Workflow: Download Data & Generate Dataset")
    df_publications = data_collection(config=config, full_query=config["Query"]["base_query_parameters"]["full_query"])

    logger.info("Workflow: Generate Author Table")
    df_authors = generate_author_table(config=config, df_publications=df_publications)

    logger.info("Workflow: Analyse Publication Metadata")
    df_publications = publication_metadata(config=config, df_publications=df_publications)

    logger.info("Workflow: Document - Term Dictionary Generation")
    build_doc_term_dict(df=df_publications, config=config)

    logger.info("Workflow: Generate TF-IDF Lists")
    tfidf_wrapper(df=df_publications, config=config)

    logger.info("Workflow: Generate TF-IDF Plotting Files")
    generate_keyword_plotting_files(config=config, df_publications=df_publications)

    logger.info("Workflow: Identify Diseases")
    identify_possible_diseases_wrapper(document_text_list=df_publications["clean_text"].tolist(), config=config)

    logger.info("Workflow: Generate Text Summaries")
    gen_summary_text_wrapper(config=config)

    if config["Workflow"]["gen_abb_dict"]:
        logger.info("Workflow: Generate Dictionary of Abbreviations")
        dict_of_abbreviations(df=df_publications, config=config)

    if config["Workflow"]["train_w2v_model"]:
        logger.info("Workflow: Train W2V Embedding")
        train_w2v_model(config=config, df=df_publications,
                        target_dir=os.path.join(config["System"]["base_dir"], config["NLP"]["skip_gram_CBOW"])
                        )

    if config["Workflow"]["train_joined_embeddings"]:
        logger.info("Workflow: Train W2V Embedding For Multiple Datasets")
        embedding_wrapper_multi_dataset(config=config, df_publications=df_publications)

    if config["Workflow"]["predict_transferable_terms"]:
        logger.info("Workflow: Predict Terms for Knowledge Transfer")
        retrospective_validation_wrapper(config=config)
        specific_term_validation(config=config)
        # TODO: Move to prediction visualisation
        # Afterwards Stop with programming!
        from Visualization.Embedding.Predictions import prediction_scatter_wrapper
        prediction_scatter_wrapper(config=config)
        # TODO: include disease
        from Visualization.Embedding.Selective_term_validation import selective_term_similarity_heatmap_wrapper
        selective_term_similarity_heatmap_wrapper(config=config)



    generate_visualisations(config=config, df_publications=df_publications, df_authors=df_authors)


def system_args_parser():
    args = sys.argv
    print(len(args), args)
    if len(args) > 1:
        if os.path.isfile(args[1]):
            main(config_path=args[1])
    else:
        print("Please provide the path to the configuration file")
        print("python main.py PATH/TO/THE/CONFIGURATION/FILE")
        print(f"you provided: {args}")


if __name__ == "__main__":
    # main(config_path="Configurations/Config_Main/config.yml")
    system_args_parser()