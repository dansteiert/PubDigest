import os, logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from misc.visualisations import set_figure_estetics
from misc.save_files import read_pickle, read_pandas_from_csv
from misc.save_figures import save_plt_figure

from Keywords.Time_intervals import get_time_intervals


def plot_publication_timeline(df_publications: pd.DataFrame, config: dict,
                              figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                              titlesize: str = "xx-large"):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Plotting Timeline and Timeintervals - Started")
    target_path = os.path.join(config["System"]["base_dir"], "Plot", "Publications_by_Year.png")
    if os.path.isfile(target_path):
        return

    # TODO: Some better way!!
    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    # <editor-fold desc="Data Generation">
    df_intervals = read_pickle(os.path.join(config["System"]["base_dir"], "Data_for_Plotting",
                                            "Publications_by_TimeIntervals.pkl"), logger=logger)
    if df_intervals is None:
        df_intervals = get_time_intervals(df=df_publications, config=config)
    # </editor-fold>

    df_intervals = df_intervals.astype({"publication year": int})
    # Publication Count WITHOUT Time Intervals
    sns.barplot(data=df_intervals, x="publication year", y="count", color=config["Visualisations"]["single_color"])
    # <editor-fold desc="figure estetics">
    plt.title("Publication Timeline")
    ax = plt.gca()
    labels = ax.get_xticklabels()
    label_text = [i.get_text() for enu, i in enumerate(labels) if
                  not i.get_text().isnumeric() or float(i.get_text()) % 5 == 0]
    label_pos = [i.get_position()[0] for enu, i in enumerate(labels) if
                 not i.get_text().isnumeric() or float(i.get_text()) % 5 == 0]
    ax.set(xticks=label_pos, xticklabels=label_text)
    plt.xticks(rotation=90)
    plt.xlabel("Publication Year")
    plt.ylabel("# Publications")
    plt.tight_layout()

    save_plt_figure(path=target_path, logger=logger)

    # </editor-fold>


def plot_publication_timeline_in_intervals(df_publications: pd.DataFrame, config: dict,
                              figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                              titlesize: str = "xx-large"):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info("Plotting Timeintervals - Started")

    target_path = os.path.join(config["System"]["base_dir"], "Plot", "Publications_by_Year_Intervals.png")
    if os.path.isfile(target_path):
        return

    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    # <editor-fold desc="Data Generation">
    df_intervals = read_pickle(os.path.join(config["System"]["base_dir"], "Data_for_Plotting",
                                            "Keywords", "Publications_by_TimeIntervals.pkl"), logger=logger)
    if df_intervals is None:
        df_intervals = get_time_intervals(df=df_publications, config=config)
    # </editor-fold>

    # Publication Count WITH Time Intervals
    interval_order = sorted(df_intervals["Time Interval"].unique().tolist(), reverse=False)
    # <editor-fold desc="Select the appropriate color palette">
    if len(interval_order) > len(config["Visualisations"]["categorial_colorfull_colors"]):
        palette = sns.color_palette(config["Visualisations"]["diverging_colors"], desat=1, n_colors=len(interval_order))

    else:
        palette = config["Visualisations"]["categorial_colorfull_colors"][:len(interval_order)]
    # </editor-fold>

    sns.barplot(data=df_intervals, x="publication year", y="count", hue="Time Interval", hue_order=interval_order,
                palette=palette,
                dodge=False)
    # <editor-fold desc="figure estetics">
    plt.legend(loc="upper left")
    plt.title("Publication Timeline")
    ax = plt.gca()
    labels = ax.get_xticklabels()
    label_text = [i.get_text() for enu, i in enumerate(labels) if
                  not i.get_text().isnumeric() or float(i.get_text()) % 5 == 0]
    label_pos = [i.get_position()[0] for enu, i in enumerate(labels) if
                 not i.get_text().isnumeric() or float(i.get_text()) % 5 == 0]
    ax.set(xticks=label_pos, xticklabels=label_text)
    plt.xticks(rotation=90)
    plt.xlabel("Publication Year")
    plt.ylabel("# Publications")
    plt.tight_layout()
    save_plt_figure(path=target_path,
                    logger=logging.getLogger(config["System"]["logging"]["logger_name"]))


def plot_publications_by_x(config: dict, df_publications: pd.DataFrame, by: str = "country", cutoff: int = 20,
                           figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    logger.info(f"Plot Publication Counts by {by} - Started")
    target_path = os.path.join(config["System"]["base_dir"], "Plot", f"Publications_at_{by}.png")
    if os.path.isfile(target_path):
        return

    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(y_axis_elements=cutoff, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>
    if df_publications is None:
        df_publications = read_pickle(os.path.join(config["System"]["base_dir"], "dataset.pkl"), logger=logger)

    df_count = df_publications.groupby(by)["affiliation id"].count().rename("count").reset_index().sort_values(by="count", ascending=False)

    # <editor-fold desc="Load data and plot as horizontal barplot">

    sns.barplot(data=df_count.head(cutoff), y=by, x="count",
                color=config["Visualisations"]["single_color"], orient="h")

    # <editor-fold desc="Figure esthetics and saving">
    plt.title(f"Number of publications by {by}")
    plt.xlabel("Publication count")
    plt.tight_layout()

    save_plt_figure(path=target_path,
                    logger=logging.getLogger(config["System"]["logging"]["logger_name"]))

    # </editor-fold>

