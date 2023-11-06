import logging, os, datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from misc.visualisations import set_figure_estetics
from misc.save_files import read_pickle
from misc.save_figures import save_plt_figure
from misc.abb_med_disease import gen_abb_med_disease


def prediction_scatter_retrospective(config: dict, query_term: str, abb: bool, med: bool, disease: bool,
                                     top_x_predictions: int = 15, novel_predictions_only: bool = False,
                                     prediction_year_marking: bool = False,
                                     figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                                     titlesize: str = "xx-large"
                                     ):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])


    set_figure_estetics(figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    base_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Embedding", abb_med,
                             f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{config['Query']['prediction_year']}_"
                             f"{config['Query']['prediction_term']}_{query_term}.pkl")
    if novel_predictions_only:
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Embedding", abb_med,
                                   f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{config['Query']['prediction_year']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}_predictions_only.png")
    else:
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Embedding", abb_med,
                                   f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{config['Query']['prediction_year']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}_retrospective.png")


    df = read_pickle(path=base_file, logger=logger)
    if df is None:
        return None

    df["Co-occurrence"] = df.apply(lambda x: True if x["co_occurrence"]> 0 else False, axis=1)
    if novel_predictions_only:
        df_groupby = df.dropna(subset=["similarity"]).groupby(by="query_term").count()
        df_groupby = df_groupby[df_groupby >= 1]
        df = df[df["query_term"].isin(set(df_groupby.index))]

        df_predictions_only = df[~df["Co-occurrence"] & df["year"].isin([int(config['Query']['prediction_year']), "before_prediction"])]
        df_groupby = df_predictions_only.groupby(by="query_term")["year"].count()
        df_groupby = df_groupby[df_groupby == 2]
        df_predictions_only = df[df["query_term"].isin(set(df_groupby.index))]
        selected_terms_dict = dict(zip(df_predictions_only["query_term"].tolist(), df_predictions_only.index.tolist()))
    else:
        selected_terms_dict = dict(zip(df["query_term"].tolist(), df.index.tolist()))
    index_term_dict = {v: k for k, v in selected_terms_dict.items()}
    sorted_idx = sorted(list(index_term_dict.keys()))[:top_x_predictions]
    selected_terms = {index_term_dict.get(i) for i in sorted_idx}
    selected_terms_dict = {i: selected_terms_dict.get(i) for i in selected_terms}
    df = df[df["query_term"].isin(selected_terms)]
    selected_terms = set(df[df["year"] != "before_prediction"]["query_term"])
    df = df[df["query_term"].isin(selected_terms)]

    df_co_occurrence = df[df["year"]=="before_prediction"].set_index("query_term")[["Co-occurrence"]].rename(columns={"Co-occurrence": "Co-occurrence\nbefore prediction"})


    df_occurrence = read_pickle(path=os.path.join(config["System"]["working_dir"], "data", config["Query"]["base_query_term"],
                                                  "Data_for_Plotting", "Embedding", query_term,
                                                  "Prediction_Term_Overlap.pkl"), logger=logger)
    df_occurrence = df_occurrence[df_occurrence.index.isin(selected_terms)]
    df_occurrence_order = pd.DataFrame(data=[{"term": i, "pos": idx} for i, idx in selected_terms_dict.items()])
    df_occurrence = df_occurrence.join(df_occurrence_order.set_index("term")).sort_values(by=["pos"])
    del df_occurrence["pos"]


    fig, (ax0, ax2, ax1) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 2, 6]})
    g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="Greys", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
    g0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelleft=True, left=True)
    g0.set_xticklabels([i.get_text() for i in g0.get_xticklabels()], rotation=90)
    g0.set_ylabel("")
    # g0.set_xlabel("Term CO-Occurrence\nbefore Prediction")


    g1 = sns.scatterplot(data=df[df["year"] != "before_prediction"].astype({"year": int}).rename(columns={"similarity": "Similarity"}),
                         x="year", y="query_term", hue="Similarity",
                         size="Co-occurrence", size_order=[True, False], #legend=False,
                         palette="coolwarm", ax=ax1)
    g1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=False, left=True)

    # g1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    # g1.set_xticks(ticks=g1.get_xticks(minor=True))
    # g1.set_xticklabels(labels=g1.get_xticklabels(which="both"), rotation=90, minor=True)
    g1.set_ylabel("")
    g1.set_xlabel("Year")
    g1.legend(bbox_to_anchor=(1, 1), loc="upper left")
    g1.set_title(#f"Dataset {config['Query']['base_query_term']} & {query_term}\n"
        f"Prediction term: {config['Query']['prediction_term']}\n"
        f"Prediction year: {config['Query']['prediction_year']}")



    g2 = sns.heatmap(df_occurrence, cbar=False, cmap="Greys", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
    g2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelleft=False, left=True)
    g2.set_xticklabels([i.get_text() for i in g2.get_xticklabels()], rotation=90)
    g2.set_ylabel("")
    g2.set_xlabel("Term occurrence\nwithin dataset")



    plt.tight_layout(w_pad=3)
    save_plt_figure(path=target_file, logger=logger)

    # </editor-fold>
def prediction_scatter(config: dict, query_term: str, abb: bool, med: bool, disease: bool,
                       top_x_predictions: int = 15, novel_predictions_only: bool = False,
                       prediction_year_marking: bool = False,
                       figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                       titlesize: str = "xx-large"
                       ):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])


    set_figure_estetics(figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    base_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Embedding", abb_med,
                             f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{config['Query']['prediction_year']}_"
                             f"{config['Query']['prediction_term']}_{query_term}.pkl")
    if novel_predictions_only:
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Embedding", abb_med,
                                   f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{config['Query']['prediction_year']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}_predictions_only.png")
    else:
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Embedding", abb_med,
                                   f"retrospective_validation_{config['NLP']['skip_gram_CBOW']}_{config['Query']['prediction_year']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}.png")


    df = read_pickle(path=base_file, logger=logger)
    if df is None:
        return None

    df["Co-occurrence"] = df.apply(lambda x: True if x["co_occurrence"]> 0 else False, axis=1)
    if novel_predictions_only:
        df_groupby = df.dropna(subset=["similarity"]).groupby(by="query_term").count()
        df_groupby = df_groupby[df_groupby >= 1]
        df = df[df["query_term"].isin(set(df_groupby.index))]

        df_predictions_only = df[~df["Co-occurrence"] & df["year"].isin([int(config['Query']['prediction_year']), "before_prediction"])]
        df_groupby = df_predictions_only.groupby(by="query_term")["year"].count()
        df_groupby = df_groupby[df_groupby == 2]
        df_predictions_only = df[df["query_term"].isin(set(df_groupby.index))]
        selected_terms_dict = dict(zip(df_predictions_only["query_term"].tolist(), df_predictions_only.index.tolist()))
    else:
        selected_terms_dict = dict(zip(df["query_term"].tolist(), df.index.tolist()))
    index_term_dict = {v: k for k, v in selected_terms_dict.items()}
    sorted_idx = sorted(list(index_term_dict.keys()))[:top_x_predictions]
    selected_terms = {index_term_dict.get(i) for i in sorted_idx}
    selected_terms_dict = {i: selected_terms_dict.get(i) for i in selected_terms}
    df = df[df["query_term"].isin(selected_terms)]
    selected_terms = set(df[df["year"] != "before_prediction"]["query_term"])
    df = df[df["query_term"].isin(selected_terms)]

    df_co_occurrence = df[df["year"]=="before_prediction"].set_index("query_term")[["Co-occurrence"]].rename(columns={"Co-occurrence": "Document Co-occurrence\nbefore prediction"})


    df_occurrence = read_pickle(path=os.path.join(config["System"]["working_dir"], "data", config["Query"]["base_query_term"],
                                                  "Data_for_Plotting", "Embedding", query_term,
                                                  "Prediction_Term_Overlap.pkl"), logger=logger)
    df_occurrence = df_occurrence[df_occurrence.index.isin(selected_terms)]
    df_occurrence_order = pd.DataFrame(data=[{"term": i, "pos": idx} for i, idx in selected_terms_dict.items()])
    df_occurrence = df_occurrence.join(df_occurrence_order.set_index("term")).sort_values(by=["pos"])
    del df_occurrence["pos"]


    fig, (ax1, ax2, ax0) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 4, 2]})
    g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="Greys", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
    # g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="coolwarm", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
    g0.tick_params(top=False, labeltop=False, bottom=True, labelbottom=False, labelleft=False, left=True)
    # g0.set_xticklabels([i.get_text() for i in g0.get_xticklabels()], rotation=90)
    g0.set_ylabel("")
    if config["Visualisations"]["with_title"]:
        g0.set_title(f"Document\nco-occurrence\nwith {config['Query']['prediction_term']}")
    g0.set_xlabel("Before prediction")


    g2 = sns.heatmap(df_occurrence, cbar=False, cmap="Greys", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
    # g2 = sns.heatmap(df_occurrence, cbar=False, cmap="coolwarm", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
    g2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=False, left=True)
    # g2.set_xticklabels([i.get_text() for i in g2.get_xticklabels()], rotation=90)
    g2.set_ylabel("")
    if config["Visualisations"]["with_title"]:
        g2.set_title("Term occurrence\nwithin dataset")

    g1 = sns.heatmap(data=df[df["year"] == config['Query']['prediction_year']].set_index("query_term").rename(columns={"similarity": "Term similarity"})[["Term similarity"]],
                     # x="year", y="query_term", hue="Similarity", #legend=False,
                     # cmap="Greys", cbar=False, annot=True, fmt=".3f", linecolor="0.5", linewidths=0.5, ax=ax1) # vmin=-1, vmax=1,
                     cmap="coolwarm", cbar=False, annot=True, fmt=".3f", linecolor="0.5", linewidths=0.5, ax=ax1) # vmin=-1, vmax=1,
    g1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=True, left=True)

    # g1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    # g1.set_xticks(ticks=g1.get_xticks(minor=True))
    # g1.set_xticklabels(labels=g1.get_xticklabels(which="both"), rotation=90, minor=True)
    g1.set_ylabel("")
    # g1.set_xlabel("Year")
    # g1.legend(bbox_to_anchor=(1, 1), loc="upper left")
    if config["Visualisations"]["with_title"]:
        g1.set_title(#f"Dataset {config['Query']['base_query_term']} & {query_term}\n"
            f"Prediction term: {config['Query']['prediction_term']}\n"
            f"Prediction year: {config['Query']['prediction_year']}")


    plt.tight_layout(w_pad=3.8, rect=(0, 0, 0.96, 1))
    # plt.show()
    save_plt_figure(path=target_file, logger=logger)

    # </editor-fold>


def prediction_scatter_wrapper(config: dict,
                               top_x_predictions: int = 15,
                               prediction_year_marking: bool = True
                               ):

    for query_term in config["Query"]["alternate_query_terms"]:
        disease = False
        for med in [True, False]:
            for abb in [True, False]:
                prediction_scatter(config=config, query_term=query_term, abb=abb, med=med, disease=disease,
                                   top_x_predictions=top_x_predictions, prediction_year_marking=prediction_year_marking,
                                   novel_predictions_only=False,
                                   figure_ratio=3/4, figure_scale=1, fontsize=10, titlesize="large")
                prediction_scatter(config=config, query_term=query_term, abb=abb, med=med, disease=disease,
                                   top_x_predictions=top_x_predictions, prediction_year_marking=prediction_year_marking,
                                   novel_predictions_only=True,
                                   figure_ratio=3/4, figure_scale=1, fontsize=10, titlesize="large")
        disease = True
        prediction_scatter(config=config, query_term=query_term, abb=abb, med=med, disease=disease,
                           top_x_predictions=top_x_predictions, prediction_year_marking=prediction_year_marking,
                           novel_predictions_only=False,
                           figure_ratio=3 / 4, figure_scale=1, fontsize=10, titlesize="large")
        prediction_scatter(config=config, query_term=query_term, abb=abb, med=med, disease=disease,
                           top_x_predictions=top_x_predictions, prediction_year_marking=prediction_year_marking,
                           novel_predictions_only=True,
                           figure_ratio=3 / 4, figure_scale=1, fontsize=10, titlesize="large")