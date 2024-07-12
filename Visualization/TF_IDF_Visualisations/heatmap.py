import logging, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from misc.visualisations import set_figure_estetics
from misc.save_files import read_pickle
from misc.save_figures import save_plt_figure
from misc.abb_med_disease import gen_abb_med_disease


from matplotlib.colors import LogNorm


def tfidf_heatmap_plotting(target_file: str, df: pd.DataFrame, config: dict, title: str, filtered: bool):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    if os.path.isfile(target_file):
        return

    # <editor-fold desc="Description">
    cols = [int(i) for i in df.columns if not (df[i] == 0).all() and i != "TF-IDF"]
    tfidf_col = "TF-IDF"

    cols = [*[i for i in range(min(cols), max(cols))],
            *[f"_{_}" for _ in range(0, len(df.columns.tolist()) // 10)], "_before", tfidf_col, "_after"
            ]
    # </editor-fold>

    # <editor-fold desc="Add Global TF IDF">
    df = df.assign(**{**{f"_{_}": 0 for _ in range(0, len(df.columns.tolist()) // 10)},
                      **{"_before": df.loc[:, tfidf_col], "_after": df.loc[:, tfidf_col]}})
    # </editor-fold>


    try:
        sns.heatmap(df[cols],
                    cmap=config["Visualisations"]["diverging_colors"], #cmap=sns.dark_palette(config["Visualisations"]["single_color"], reverse=False, as_cmap=True)
                    norm=LogNorm(),
                    xticklabels=True, yticklabels=filtered) #
    except Exception as e:
        logger.fatal(
            f"{e} - TFIDF Heatmap -  {config['Visualisations']['single_color']}, {target_file} - {df.describe()}")
        exit()

    if config["Visualisations"]["with_title"]:
        plt.title(title)

    # <editor-fold desc="Remove X Lables to separate the Globale TF-IDF">
    ax = plt.gca()
    labels = ax.get_xticklabels()
    try:
        label_text = [i.get_text() for enu, i in enumerate(labels) if not i.get_text().isnumeric() or int(i.get_text()) % 5 == 0]
        label_pos = [i.get_position()[0] for enu, i in enumerate(labels) if not i.get_text().isnumeric() or int(i.get_text()) % 5 == 0]
        label_pos = [i for i, j in zip(label_pos, label_text) if not "_" in j]
        label_text = [i for i in label_text if not "_" in i]
        ax.set(xticks=label_pos, xticklabels=label_text)

    except Exception as e:
        logger.error(f"{e} - TFIDF Heatmap - Assign Label Positions")
    # </editor-fold>


    if filtered:
        plt.grid(b=True, which="major", axis="y")

    plt.tight_layout(rect=(0, 0, 0.95, 1))
    save_plt_figure(path=target_file, logger=logger)


def tfidf_heatmap(config: dict, n_gram: int = 1, med: bool = False, abb: bool = False, disease: bool = False,
                  cutoff: int = 40,
                  figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10, titlesize: str = "xx-large"):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)


    target_file_filtered_sorted_global = os.path.join(config["System"]["base_dir"], "Plot", "Keywords",
                                                      f"{n_gram}_gram", abb_med,
                                                      f"tfidf_{n_gram}_gram{abb_med}_filtered_global_sort.png")
    target_file_filtered = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med,
                                        f"tfidf_{n_gram}_gram{abb_med}_filtered.png")
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med,
                               f"tfidf_{n_gram}_gram{abb_med}.png")

    # <editor-fold desc="Load data">
    df = read_pickle(os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                                  f"tfidf_{n_gram}_gram{abb_med}_filterable.pkl"), logger=logger)
    if df is None:
        logger.error(f"tfidf_{n_gram}_gram{abb_med}.csv - File does not exist since there are no entries")
        return
    # </editor-fold>
    df = df.rename(columns={"sum": "TF-IDF"})
    column_list = [*sorted([i for i in df.columns if i not in [config["NLP"]["tfidf_filter_criterion"], "idf", "TF-IDF"] and i > config["Visualisations"]["heatmap_start_year"]]), "TF-IDF"]
    df = df.sort_values(by=column_list, ascending=True)
    # column_list[-1] = "TF-IDF"


    # <editor-fold desc="Call TF-IDF Plot Generation Functions">
    if not os.path.isfile(target_file):
        set_figure_estetics(figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)

        tfidf_heatmap_plotting(target_file=target_file, df=df[column_list], config=config,
                               title=f"TF-IDF Values by Year\n{n_gram}_gram{'; Medical' if med else ''}"
                                     f"{'; Abbreviations' if abb else ''}{'; Disease' if disease else ''}",
                               filtered=False)

    if not os.path.isfile(target_file_filtered):
        # df_filtered = df.nlargest(cutoff, columns=[config["NLP"]["tfidf_filter_criterion"]])
        df_filtered = df.nlargest(cutoff, columns=["TF-IDF"])
        df_filtered = df_filtered[column_list]
        df_filtered = df_filtered.sort_values(by=column_list, ascending=True)

        set_figure_estetics(y_axis_elements=df_filtered.shape[0], figure_ratio=figure_ratio,
                            figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)

        tfidf_heatmap_plotting(target_file=target_file_filtered, df=df_filtered, config=config,
                               title=f"TF-IDF Values by Year\n{n_gram}_gram{'; Medical' if med else ''}"
                                     f"{'; Abbreviations' if abb else ''}{'; Disease' if disease else ''}",
                               filtered=True)

    if not os.path.isfile(target_file_filtered_sorted_global):
        # df_filtered = df.nlargest(cutoff, columns=[config["NLP"]["tfidf_filter_criterion"]])
        df_filtered = df.nlargest(cutoff, columns=["TF-IDF"]) #
        df_filtered = df_filtered[column_list]
        df_filtered = df_filtered.sort_values(by=["TF-IDF"], ascending=False)

        set_figure_estetics(y_axis_elements=df_filtered.shape[0], figure_ratio=figure_ratio,
                            figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)

        tfidf_heatmap_plotting(target_file=target_file_filtered_sorted_global, df=df_filtered, config=config,
                               title=f"TF-IDF values by year\n{n_gram}-gram{'; Medical' if med else ''}"
                                     f"{'; Abbreviations' if abb else ''}{'; Clinical features' if disease else ''}",
                               filtered=True)
    # </editor-fold>

