import pandas as pd
import os, logging
import seaborn as sns
import matplotlib.pyplot as plt


from misc.abb_med_disease import gen_abb_med_disease
from misc.save_files import read_pickle

def gen_tsne_representation(config: dict, abb: bool, med: bool, disease: bool, dataset_a: str, dataset_b: set):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

    # target_file = os.path.join(config["System"]["working_dir"], "data", dataset_a, "Data_for_Plotting", "Embedding",
    #                            dataset_b, f"Embedding_tsne{abb_med}.pkl")
    target_file = os.path.join(config["System"]["working_dir"], "data", dataset_a, "Data_for_Plotting", "Embedding",
                               dataset_b, f"Embedding_LLE{abb_med}.pkl")
    df = read_pickle(target_file, logger=logger)
    df_occurrence = read_pickle(
        path=os.path.join(config["System"]["working_dir"], "data", dataset_a,
                          "Data_for_Plotting", "Embedding", dataset_b,
                          "Prediction_Term_Overlap.pkl"), logger=logger)
    df_joined = pd.merge(left=df, right=df_occurrence, left_index=True, right_index=True, how="left")
    df_joined["Occurence"] = df_joined.apply(lambda x: occurence_dataset(row=x, target="CTEPH", config=config), axis=1)
    df_joined = df_joined[df_joined["Occurence"].isin(["Target", "VT", "CTEPH", "Both"])]

    target_terms = set(["riociguat", "macitentan", "bosentan", "sildenafil", "iloprost", "rivaroxaban", "caplacizumab",
                        "benzarone", "desirudin", "abciximab", "cilostazol", "dipyridamole", "carfilzomib",
                        "metformin", "etanercept", "thalidomide", "tamoxifen", "ponatinib"])

    ax = sns.scatterplot(data=df_joined, x="x", y="y", hue="Occurence")

    target_coord = df_joined.loc["CTEPH"].to_list()[:2]

    # ax.xlim(target_coord[0] - 5, target_coord[0] + 5)
    # ax.ylim(target_coord[1] -  5, target_coord[1] + 5)
    for term, pos in df_joined[df_joined.index.isin(target_terms)][["x", "y"]].iterrows():
        ax.annotate(term, pos)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()

    plt.show()



def occurence_dataset(row, target: str, config: dict):

    if row.name == target:
        return "Target"
    if row["CTEPH"]:
        if row["VT"]:
            return "Both"
        return "CTEPH"
    if row["VT"]:
        return "VT"
    return "Undetermined"

def tsne_representation_wrapper(config: dict):
    for query_term in config["Query"]["alternate_query_terms"]:
        disease = False
        for med in [True, False]:
            for abb in [True, False]:
                gen_tsne_representation(config=config, abb=abb, med=med, disease=disease, dataset_a=config["Query"]["base_query_term"],dataset_b=query_term)
        disease = True
        gen_tsne_representation(config=config, abb=abb, med=med, disease=disease,
                                dataset_a=config["Query"]["base_query_term"], dataset_b=query_term)
