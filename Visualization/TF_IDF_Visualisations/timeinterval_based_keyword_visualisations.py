import os, logging
import pandas as plt
import matplotlib.pyplot as plt
import seaborn as sns

from misc.visualisations import set_figure_estetics
from misc.abb_med_disease import gen_abb_med_disease
from misc.save_files import read_pickle
from misc.save_figures import save_plt_figure



def plot_keywords_as_mosaik(config: dict, n_gram: int, abb: bool, med: bool, disease: bool, cutoff: int = 100,
                            figure_ratio: float = 2 / 1, figure_scale: float = 1,
                            fontsize: int = 10, titlesize: str = "x-large"):
    '''
    Build a plot of keywords and their occurance within different time intervals
    :param cutoff: how many keywords are shown
    :return: None -> Plot made in the Plot Folder
    '''

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    # <editor-fold desc="Load data">
    abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med,
                               f"{n_gram}_gram{abb_med}_Keyword_as_Mosaik.png")
    data_file = os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords",  f"{n_gram}",
                                  f"{n_gram}_gram{abb_med}_Keyword_as_Mosaik.pkl")
    if os.path.isfile(target_file):
        return


    df = read_pickle(data_file, logger=logger)

    if df is None:
        logging.error(f"Mosaik Diagram - No such file exist - potentially no occurrences in the dataset - {data_file}")
        return
    # </editor-fold>

    # <editor-fold desc="select top x(cutoff) words and determine which years are present">
    df = df.head(cutoff)
    col_list = sorted([i for i in df.columns if "bottom" not in i])
    # </editor-fold>

    # <editor-fold desc="sort dataframe in reverse">
    df = df[::-1]
    # </editor-fold>

    # <editor-fold desc="select appropriate colors">
    if len(col_list) > len(config["categorial_colorfull_colors"]):
        # palette = sns.color_palette(palette="Set2", n_colors=len(col_list), desat=1)
        palette = sns.color_palette(config["diverging_colors"], desat=1, n_colors=len(col_list))
    else:
        palette = config["categorial_colorfull_colors"][:len(col_list)]
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(y_axis_elements=df.shape[0], figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    # <editor-fold desc="Stacked barplot horizontal for each term">
    for index, (i, color) in enumerate(zip(col_list, palette)):
        if index == 0:
            plt.barh(y=df.index, width=df[i], label=i, edgecolor="w", color=color)
        else:
            plt.barh(y=df.index, left=df[f"{i} bottom"], width=df[i], label=i, edgecolor="w", color=color)
    # </editor-fold>

    # <editor-fold desc="Figure estetitcs and saving">
    plt.legend()
    plt.xlabel("Timeinterval TF-IDF score")
    plt.ylabel("")
    plt.title(f"Timeinterval TF-IDF score\n{n_gram}_gram [{abb_med}")
    plt.tight_layout()
    save_plt_figure(path=target_file, logger=logger)
    # </editor-fold>


# TODO: continue here!!
def plot_keywords_as_venn_diagram(config: dict, cutoff: int = 100, n_gram: int = 1,
                                  med: bool = False, abb: bool = False,
                                  figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10, titlesize: str = "xx-large"):
    '''
    Build a plot of keywords and their occurance within different time intervals
    :param cutoff: how many keywords are shown
    :return: None -> Plot made in the Plot Folder
    '''
    # https://caleydo.org/tools/upset/
    # reused Idea!

    # <editor-fold desc="Load data">
    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    target_file = os.path.join(config["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med, f"{n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.png")
    if os.path.isfile(target_file):
        return
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    df = read_pandas_from_csv(
        os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                     f"{n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.csv"))
    if df is None:
        data_keywords_by_timeinterval(config=config)
        df = read_pandas_from_csv(
            os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                         f"{n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.csv"))
        if df is None:
            logging.error \
                (f"Venn Diagram - {n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.csv - No such file exist - potentially no occurances in the dataset")
            return
    # </editor-fold>

    # <editor-fold desc="select top x(cutoff) words and determine which years are present">
    df = df.head(cutoff)
    col_list = sorted([i for i in df.columns if "bottom" not in i])
    # </editor-fold>

    # <editor-fold desc="reverse ordering of dataframe">
    df = df.sort_values(by=col_list)
    # </editor-fold>

    # <editor-fold desc="select appropriate colors">
    if len(col_list) > len(config["categorial_colorfull_colors"]):
        # palette = sns.color_palette(palette="Set2", n_colors=len(col_list), desat=1)
        palette = sns.color_palette(config["diverging_colors"], desat=1, n_colors=len(col_list))
    else:
        palette = config["categorial_colorfull_colors"][:len(col_list)]
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["base_dir"], "Plot" ,"Keywords", f"{n_gram}_gram", abb_med ,), exist_ok=True)
    set_figure_estetics(y_axis_elements=df.shape[0], figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    # <editor-fold desc="Stacked horizontal barplot of term usage">
    for index, (i, color) in enumerate(zip(col_list, palette)):
        if index == 0:
            plt.barh(y=df.index, width=df[i], label=i, edgecolor="w", color=color)
        else:
            plt.barh(y=df.index, left=df[f"{i} bottom"], width=df[i], label=i, edgecolor="w", color=color)
    # </editor-fold>

    # <editor-fold desc="Figure esthetics and saving">
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="medium")
    # plt.xticks(rotation=90)
    plt.xlabel("Normalized TF-IDF Score")
    plt.ylabel("")
    plt.title(
        f"TF-IDF score by Time Intervals\n{n_gram}_gram{'; Medical' if med else ''}{'; Abbreviations' if abb else ''}")
    plt.tight_layout()
    plt.savefig(target_file,
                dpi=400, transparent=True)
    plt.close('all')
    # </editor-fold>



def plot_keywords_by_timeinterval(config: dict, cutoff: int = 35, n_gram: int = 1, med: bool = False,
                                  abb: bool = False,
                                  figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                                  titlesize: str = "xx-large"):
    # <editor-fold desc="Load data">

    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    target_folder = os.path.join(config["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med)

    os.makedirs(target_folder, exist_ok=True)

    df = read_pandas_from_csv(os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                                           f"{n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv"))
    if df is None:
        data_keywords_by_timeinterval(config=config)
        df = read_pandas_from_csv(os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                                               f"{n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv"))
        if df is None:
            logging.error \
                (f"keywords by Timeinterval - {n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv - No such file exist - potentially no occurances in the dataset")
            return
    # </editor-fold>

    years = sorted(df.columns)
    # <editor-fold desc="select appropriate colors">
    if len(years) > len(config["categorial_colorfull_colors"]):
        palette = sns.color_palette(config["diverging_colors"], desat=1, n_colors=len(years))
    else:
        palette = config["categorial_colorfull_colors"][:len(years)]
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["base_dir"], "Plot" ,"Keywords", f"{n_gram}_gram", abb_med ,), exist_ok=True)
    set_figure_estetics(y_axis_elements=cutoff, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    for y, color in zip(years, palette):
        target_file = os.path.join(target_folder, f"{n_gram}_gram{abb_med}_Keyword_{y}.png")
        if os.path.isfile(target_file):
            continue

        # <editor-fold desc="Horizontal barplot of a Timeinterval with the top x(cutoff) terms">
        df_temp = df.sort_values(by=y, ascending=False).head(cutoff)
        sns.barplot(data=df_temp, y=df_temp.index, x=y, color=color)
        # </editor-fold>

        # <editor-fold desc="Figure esthetics">
        plt.xlabel("TF-IDF Score")
        plt.title(
            f"TD-IDF score in Timeinterval: {y}\n{n_gram}_gram{'; Medical' if med else ''}{'; Abbreviations' if abb else ''}")
        plt.tight_layout()
        plt.savefig(target_file, dpi=400, transparent=True)
        plt.close('all')
        # </editor-fold>