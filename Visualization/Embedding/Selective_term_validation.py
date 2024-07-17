import logging, os, datetime
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import seaborn as sns

from misc.save_files import read_pickle
from misc.visualisations import set_figure_estetics
from misc.save_figures import save_plt_figure


def selective_term_similarity_heatmap(config: dict, query_term: str,
                       figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                       titlesize: str = "xx-large"
                       ):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])


    set_figure_estetics(figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    base_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Embedding",
                                   f"specific_term_validation_{config['NLP']['skip_gram_CBOW']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}.pkl")




    df = read_pickle(path=base_file, logger=logger)
    if df is None:
        return None

    max_entries = 6

    df["Co-occurrence"] = df.apply(lambda x: True if x["co_occurrence"]> 0 else False, axis=1)
    df = df[df["query_term"].isin(set(df[df["year"] != "before_prediction"]["query_term"]))]
    df = df.sort_values(by="similarity", ascending=False)
    df = df.drop_duplicates(subset=["query_term", "year"], keep="first")
    selected_terms_dict = dict(zip(reversed(df["query_term"].tolist()), reversed(df.index.tolist())))
    df_occurrence = read_pickle(
        path=os.path.join(config["System"]["working_dir"], "data", config["Query"]["base_query_term"],
                          "Data_for_Plotting", "Embedding", query_term,
                          "Prediction_Term_Overlap_specific_term_validation.pkl"), logger=logger)
    df_occurrence = df_occurrence[df_occurrence.index.isin(set(df["query_term"]))]
    v_min = df["similarity"].head(12*max_entries).min()
    v_max = df["similarity"].max()

    max_entries = 6
    for idx, terms in enumerate([*config["Query"]["terms_for_embedding_evaluation"], [i for j in config["Query"]["terms_for_embedding_evaluation"] for i in j]]):
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Embedding",
                                   f"specific_term_validation_{config['NLP']['skip_gram_CBOW']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}_group_{idx}.png")
        if idx ==3:
            max_entries = 3* max_entries
        if idx == 0:
            selected_terms = set(terms)
            df_temp = df[df["query_term"].isin(selected_terms)]
        else:
            selected_terms = set(terms)
            transferable_terms_only = set(df_occurrence[~df_occurrence[config["Query"]["base_query_abb"]]].index)
            selected_terms = set([i for i in selected_terms if i in transferable_terms_only])
            df_temp = df[df["query_term"].isin(selected_terms) & (~df["Co-occurrence"])]

        if df_temp.shape[0] == 0:
            continue
        df_temp = df_temp.sort_values(by="similarity", ascending=False)
        selected_terms = set(df_temp["query_term"].head(max_entries))

        df_co_occurrence = df_temp[(df_temp["year"]=="before_prediction") & (df_temp["query_term"].isin(selected_terms))].set_index("query_term")[["Co-occurrence"]].rename(columns={"Co-occurrence": "Document Co-occurrence\nbefore prediction"})



        df_occurrence_temp = df_occurrence[df_occurrence.index.isin(selected_terms)]
        df_occurrence_order = pd.DataFrame(data=[{"term": i, "pos": idx} for i, idx in selected_terms_dict.items() if i in selected_terms])
        df_occurrence_temp = df_occurrence_temp.join(df_occurrence_order.set_index("term")).sort_values(by=["pos"])
        del df_occurrence_temp["pos"]


        fig, (ax1, ax2, ax0) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 4, 2]}, **{"figsize": (6,2)})
        g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="Blues", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
        # g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="coolwarm", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
        g0.tick_params(top=False, labeltop=False, bottom=True, labelbottom=False, labelleft=False, left=True)
        # g0.set_xticklabels([i.get_text() for i in g0.get_xticklabels()], rotation=90)
        g0.set_ylabel("")
        if config["Visualisations"]["with_title"]:
            g0.set_title(f"Document\nco-occurrence\nwith {config['Query']['prediction_term']}")
        g0.set_xlabel("Before prediction")


        g2 = sns.heatmap(df_occurrence_temp, cbar=False, cmap="Blues", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
        # g2 = sns.heatmap(df_occurrence, cbar=False, cmap="coolwarm", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
        g2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=False, left=True)
        # g2.set_xticklabels([i.get_text() for i in g2.get_xticklabels()], rotation=90)
        g2.set_ylabel("")
        if config["Visualisations"]["with_title"]:
            g2.set_title("Term occurrence\nwithin dataset")

        g1 = sns.heatmap(data=df_temp[df_temp["year"] == datetime.date.today().year].set_index("query_term").rename(columns={"similarity": "Term similarity"})[["Term similarity"]].head(max_entries),
                             # x="year", y="query_term", hue="Similarity", #legend=False,
                        # cmap="Greys", cbar=False, annot=True, fmt=".3f", linecolor="0.5", linewidths=0.5, ax=ax1) # vmin=-1, vmax=1,
                        cmap="coolwarm", cbar=False, annot=True, fmt=".3f", linecolor="0.5", linewidths=0.5, ax=ax1,
                         vmin=v_min, vmax=v_max)
        g1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=True, left=True)
        g1.tick_params(axis="y", rotation=0)

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
def selective_term_similarity_heatmap_publication(config: dict, query_term: str,
                       figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                       titlesize: str = "xx-large"
                       ):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])


    set_figure_estetics(figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    base_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Embedding",
                                   f"specific_term_validation_{config['NLP']['skip_gram_CBOW']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}.pkl")




    df = read_pickle(path=base_file, logger=logger)
    if df is None:
        return None

    max_entries = 6

    df["Co-occurrence"] = df.apply(lambda x: True if x["co_occurrence"]> 0 else False, axis=1)
    df = df[df["query_term"].isin(set(df[df["year"] != "before_prediction"]["query_term"]))]
    df = df.sort_values(by="similarity", ascending=False)
    df = df.drop_duplicates(subset=["query_term", "year"], keep="first")
    selected_terms_dict = dict(zip(reversed(df["query_term"].tolist()), reversed(df.index.tolist())))
    df_occurrence = read_pickle(
        path=os.path.join(config["System"]["working_dir"], "data", config["Query"]["base_query_term"],
                          "Data_for_Plotting", "Embedding", query_term,
                          "Prediction_Term_Overlap_specific_term_validation.pkl"), logger=logger)
    df_occurrence = df_occurrence[df_occurrence.index.isin(set(df["query_term"]))]
    v_min = df["similarity"].head(12*max_entries).min()
    v_max = df["similarity"].max()

    max_entries = 6
    for idx, terms in enumerate([*config["Query"]["terms_for_embedding_evaluation"], [i for j in config["Query"]["terms_for_embedding_evaluation"] for i in j]]):
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Embedding",
                                   f"specific_term_validation_{config['NLP']['skip_gram_CBOW']}_"
                                   f"{config['Query']['prediction_term']}_{query_term}_group_{idx}_publication.png")
        if idx ==3:
            max_entries = 3* max_entries
        if idx == 0:
            selected_terms = set(terms)
            df_temp = df[df["query_term"].isin(selected_terms)]
        else:
            selected_terms = set(terms)
            transferable_terms_only = set(df_occurrence[~df_occurrence[config["Query"]["base_query_abb"]]].index)
            selected_terms = set([i for i in selected_terms if i in transferable_terms_only])
            df_temp = df[df["query_term"].isin(selected_terms) & (~df["Co-occurrence"])]

        if df_temp.shape[0] == 0:
            continue
        df_temp = df_temp.sort_values(by="similarity", ascending=False)
        selected_terms = set(df_temp["query_term"].head(max_entries))

        # df_co_occurrence = df_temp[(df_temp["year"]=="before_prediction") & (df_temp["query_term"].isin(selected_terms))].set_index("query_term")[["Co-occurrence"]].rename(columns={"Co-occurrence": "Document Co-occurrence\nbefore prediction"})



        df_occurrence_temp = df_occurrence[df_occurrence.index.isin(selected_terms)]
        df_occurrence_order = pd.DataFrame(data=[{"term": i, "pos": idx} for i, idx in selected_terms_dict.items() if i in selected_terms])
        df_occurrence_temp = df_occurrence_temp.join(df_occurrence_order.set_index("term")).sort_values(by=["pos"])
        del df_occurrence_temp["pos"]


        # fig, (ax1, ax2, ax0) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 4, 2]}, **{"figsize": (6,2)})
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 4]}, **{"figsize": (6,2)})
        # g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="Blues", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
        # # g0 = sns.heatmap(data=df_co_occurrence, cbar=False, cmap="coolwarm", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax0)
        # g0.tick_params(top=False, labeltop=False, bottom=True, labelbottom=False, labelleft=False, left=True)
        # # g0.set_xticklabels([i.get_text() for i in g0.get_xticklabels()], rotation=90)
        # g0.set_ylabel("")
        # if config["Visualisations"]["with_title"]:
        #     g0.set_title(f"Document\nco-occurrence\nwith {config['Query']['prediction_term']}")
        # g0.set_xlabel("Before prediction")


        g2 = sns.heatmap(df_occurrence_temp, cbar=False, cmap="Blues", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
        # g2 = sns.heatmap(df_occurrence, cbar=False, cmap="coolwarm", linecolor="0.5", linewidths=0.5, vmin=0, vmax=1, ax=ax2)
        g2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=False, left=True)
        # g2.set_xticklabels([i.get_text() for i in g2.get_xticklabels()], rotation=90)
        g2.set_ylabel("")
        if config["Visualisations"]["with_title"]:
            g2.set_title("Term occurrence\nwithin dataset")

        g1 = sns.heatmap(data=df_temp[df_temp["year"] == datetime.date.today().year].set_index("query_term").rename(columns={"similarity": "Term similarity"})[["Term similarity"]].head(max_entries),
                             # x="year", y="query_term", hue="Similarity", #legend=False,
                        # cmap="Greys", cbar=False, annot=True, fmt=".3f", linecolor="0.5", linewidths=0.5, ax=ax1) # vmin=-1, vmax=1,
                        cmap="coolwarm", cbar=False, annot=True, fmt=".3f", linecolor="0.5", linewidths=0.5, ax=ax1,
                         vmin=v_min, vmax=v_max)
        g1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, labelleft=True, left=True)
        g1.tick_params(axis="y", rotation=0)
        make_axes_area_auto_adjustable(ax=ax1, pad=0.15, adjust_dirs=["left"])

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


def selective_term_similarity_heatmap_wrapper(config: dict):

    for query_term in config["Query"]["alternate_query_terms"]:
        selective_term_similarity_heatmap(config=config, query_term=query_term)
        selective_term_similarity_heatmap_publication(config=config, query_term=query_term)