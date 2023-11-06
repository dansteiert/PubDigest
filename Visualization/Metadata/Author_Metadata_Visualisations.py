import os, logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from misc.visualisations import set_figure_estetics
from misc.save_files import read_pickle
from misc.save_figures import save_plt_figure

def plot_publications_by_author(config: dict, df_authors: pd.DataFrame, cutoff: int = 40,
                                figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10):
    # TODO: cutoff in Configs
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    logger.info("Plotting Publication Counts - Start")

    if df_authors is None:
        read_pickle(os.path.join(config["System"]["base_dir"], "authors.pkl"), logger=logger)
    # <editor-fold desc="Select the top x publishing authors">
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(y_axis_elements=cutoff, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>

    translation_dict = {"publication_count": "All Publications", "first_author": "First Authored Publications",
                        "last_author": "Last Authored Publications"}
    for i in ["publication_count", "first_author", "last_author"]:
        target_file = os.path.join(config["System"]["base_dir"], "Plot", "Publications", f"{translation_dict.get(i)}_by_Author.png")
        if os.path.isfile(target_file):
            continue

        df_temp = df_authors.sort_values(by=i, ascending=False).head(cutoff)

        # <editor-fold desc="Plot horizontal bars of publication count">

        sns.barplot(data=df_temp, y="reference name", x=i, color=config["Visualisations"]["single_color"])
        # </editor-fold>

        # <editor-fold desc="Figure esthetics">
        plt.title(f"{translation_dict.get(i)}")
        plt.ylabel("Author Name")
        plt.xlabel("Publication Count")
        # plt.xticks(rotation=90)
        plt.tight_layout()

        save_plt_figure(path=target_file, logger=logger)
    # </editor-fold>
    logger.info("Plotting Publication Counts - Finished")


def plot_citations_by_author(config: dict, df_authors: pd.DataFrame, cutoff: int = 40,
                             figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Plotting Citation Counts - Start")
    if df_authors is None:
        df_authors = read_pickle(os.path.join(config["System"]["base_dir"], "authors.pkl"), logger=logger)

    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(y_axis_elements=cutoff, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>

    translation_dict = {"citation count": "Citations for All Publications", "citation count - first author": "Citations for First Authored Publications",
                        "citation count - last author": "Citations for Last Authored Publications"}
    for i in ["citation count", "citation count - first author", "citation count - last author"]:
        if "citation" in i:
            target_file = os.path.join(config["System"]["base_dir"], "Plot", "Citations", f"{translation_dict.get(i)}_by_Author.png")
            if os.path.isfile(target_file):
                continue
            df_temp = df_authors.sort_values(by=i, ascending=False).head(cutoff)
            # <editor-fold desc="Horizontal barplot sorted for the most citations by an author">
            sns.barplot(data=df_temp, y="reference name", x=i, color=config["Visualisations"]["single_color"])
            # </editor-fold>

            # <editor-fold desc="Figure esthetics">
            plt.title(f"{translation_dict.get(i)}")
            plt.ylabel("Author Name")
            plt.xlabel("Citation Count")
            # plt.xticks(rotation=90)
            plt.tight_layout()
            save_plt_figure(path=target_file, logger=logger)
    # </editor-fold>
    logger.info("Plotting Citation Counts - Finished")


