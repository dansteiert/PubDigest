import os, sys, logging
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from misc.save_files import read_pandas_from_csv
from misc.visualisations import set_figure_estetics


def plot_keywords_as_mosaik(config: dict, cutoff: int = 100, n_gram: int = 1, med: bool = False,
                            abb: bool = False,
                            figure_ratio: float = 2 / 1, figure_scale: float = 1,
                            fontsize: int = 10, titlesize: str = "xx-large"):
    '''
    Build a plot of keywords and their occurance within different time intervals
    :param cutoff: how many keywords are shown
    :return: None -> Plot made in the Plot Folder
    '''
    
    # <editor-fold desc="Load data">
    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med, f"{n_gram}_gram{abb_med}_Keyword_as_Mosaik.png")
    if os.path.isfile(target_file):
        return
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])


    df = read_pandas_from_csv(
            os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords",  f"{n_gram}", f"{n_gram}_gram{abb_med}_Keyword_as_Mosaik.csv"),
            logger=logger, nrows=cutoff)
    if df is None:
        logging.error(f"Mosaik Diagram - {n_gram}_gram{abb_med}_Keyword_as_Mosaik.csv - No such file exist - potentially no occurances in the dataset")
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
    if len(col_list) > len(config["Visualizations"]["categorial_colorfull_colors"]):
        # palette = sns.color_palette(palette="Set2", n_colors=len(col_list), desat=1)
        palette = sns.color_palette(config["Visualizations"]["diverging_colors"], desat=1, n_colors=len(col_list))
    else:
        palette = config["Visualizations"]["categorial_colorfull_colors"][:len(col_list)]
    # </editor-fold>
    
    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["System"]["base_dir"], "Plot","Keywords", f"{n_gram}_gram", abb_med,), exist_ok=True)
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
    # plt.xticks(rotation=90)
    plt.xlabel("Normalized TF-IDF score")
    plt.ylabel("")
    plt.title(
        f"Proportional TF-IDF score by Time Intervals\n{n_gram}_gram{'; Medical' if med else ''}{'; Abbreviations' if abb else ''}")
    plt.tight_layout()
    plt.savefig(target_file, dpi=400, transparent=True)
    plt.close('all')
    # </editor-fold>


def plot_keyword_total_usage(config: dict,  cutoff: int = 100, n_gram: int = 1, med: bool = False,
                             abb: bool = False, figure_ratio: float = 2 / 1,
                             figure_scale: float = 1, fontsize: int = 10, titlesize: str = "xx-large"):
    '''
    Build a plot of keywords and their occurance within different time intervals
    :param cutoff: how many keywords are shown
    :return: None -> Plot made in the Plot Folder
    '''
    
    # <editor-fold desc="Load data">
    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med, f"{n_gram}_gram{abb_med}_Keyword_usage.png")
    if os.path.isfile(target_file):
        return
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])


    df = read_pandas_from_csv(path=os.path.join(config["System"]["base_dir"], "Data_for_Plotting",
                                                "Keywords",  f"{n_gram}",
                                                f"tfidf_{n_gram}_gram{abb_med}_filterable.csv"),
                              nrows=cutoff, logger=logger)
    if df is None:
        logging.error(f"Mosaik Diagram - tfidf_{n_gram}_gram{abb_med}_filterable.csv - No such file exist - potentially no occurances in the dataset")
        return
    # </editor-fold>
    
    
    # <editor-fold desc="Horizontal barplor of Term usage by TF-IDF">
    
    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["System"]["base_dir"], "Plot","Keywords", f"{n_gram}_gram", abb_med,), exist_ok=True)
    set_figure_estetics(y_axis_elements=df.shape[0], figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>
    
    sns.barplot(data=df, x="sum", y=df.index, color=config["Visualizations"]["single_color"])
    # </editor-fold>
    
    # <editor-fold desc="Figure esthetics and saving">
    plt.ylabel("")
    plt.xlabel("TF-IDF Score")
    plt.title(f"Global TF-IDF score\n{n_gram}_gram{'; Medical' if med else ''}{'; Abbreviations' if abb else ''}")
    plt.tight_layout()
    plt.savefig(target_file, dpi=400, transparent=True)
    plt.close('all')
    # </editor-fold>


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
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med, f"{n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.png")
    if os.path.isfile(target_file):
        return
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    df = read_pandas_from_csv(
            os.path.join(config["System"]["base_dir"], "Data_for_Plotting",
                         "Keywords", f"{n_gram}",
                         f"{n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.csv"),
        logger=logger)
    if df is None:
        logging.error(f"Venn Diagram - {n_gram}_gram{abb_med}_Keyword_as_Venn_Diagram.csv - No such file exist - potentially no occurances in the dataset")
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
    if len(col_list) > len(config["Visualizations"]["categorial_colorfull_colors"]):
        # palette = sns.color_palette(palette="Set2", n_colors=len(col_list), desat=1)
        palette = sns.color_palette(config["Visualizations"]["diverging_colors"], desat=1, n_colors=len(col_list))
    else:
        palette = config["Visualizations"]["categorial_colorfull_colors"][:len(col_list)]
    # </editor-fold>
    
    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["System"]["base_dir"], "Plot","Keywords", f"{n_gram}_gram", abb_med,), exist_ok=True)
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
    target_folder = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med)

    os.makedirs(target_folder, exist_ok=True)

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    df = read_pandas_from_csv(os.path.join(config["System"]["base_dir"],
                                           "Data_for_Plotting", "Keywords", f"{n_gram}",
                         f"{n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv"),
                              logger=logger)
    if df is None:
        logging.error(f"keywords by Timeinterval - {n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv - No such file exist - potentially no occurances in the dataset")
        return
    # </editor-fold>
    
    years = sorted(df.columns)
    # <editor-fold desc="select appropriate colors">
    if len(years) > len(config["Visualizations"]["categorial_colorfull_colors"]):
        palette = sns.color_palette(config["Visualizations"]["diverging_colors"], desat=1, n_colors=len(years))
    else:
        palette = config["Visualizations"]["categorial_colorfull_colors"][:len(years)]
    # </editor-fold>
    
    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["System"]["base_dir"], "Plot","Keywords", f"{n_gram}_gram", abb_med,), exist_ok=True)
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


def plot_delta_keyword_frequency(base_dir: str = None, n_gram: int = 1, cutoff: int = 100, med: bool = False,
                                 abb: bool = False):
    return None
    # <editor-fold desc="Load data">
    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    if not os.path.isdir(os.path.join(base_dir, "Plot", "Keywords", f"{n_gram}_gram", abb_med)):
        if not os.path.isdir(os.path.join(base_dir, "Plot", "Keywords", f"{n_gram}_gram")):
            os.mkdir(os.path.join(base_dir, "Plot", "Keywords", f"{n_gram}_gram"))
        if abb_med != "":
            os.mkdir(os.path.join(base_dir, "Plot", "Keywords", f"{n_gram}_gram", abb_med))
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Keywords",  f"{n_gram}", f"tfidf_{n_gram}_gram{abb_med}.csv")):
        df = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Keywords",  f"{n_gram}", f"tfidf_{n_gram}_gram{abb_med}.csv"),
                         index_col=0)
    else:
        logging.error(f"delta keyword frequency - tfidf_{n_gram}_gram{abb_med}.csv - No such file exist - potentially no occurances in the dataset")
        return
    # </editor-fold>
    
    # Normalize with publication by year
    df_paper = pd.read_csv(os.path.join(base_dir, "Paper_Table.csv"), index_col=0)
    df_paper["year"] = df_paper.apply(lambda x: int(x["publication_date"][:4]), axis=1)
    df_groupby = df_paper.groupby(by="year")["publication_date"].apply(len)
    
    ## sort and keep only the most common keywords
    df = df.head(cutoff)
    
    # normalize scores
    for year, num_docs in zip(df_groupby.index, df_groupby.values):
        df[str(year)] = df[str(year)] / num_docs
    
    col_list = sorted([i for i in df.columns if "frequency" not in i])
    df_delta = pd.DataFrame()
    df_delta["index"] = df.index
    df_delta = df_delta.set_index("index")
    for index, i in enumerate(col_list):
        if index == 0:
            df_delta[i] = 0
            continue
        df_delta[i] = df[col_list[index - 1]] - df[i]
    
    df_melt = df_delta.melt(value_vars=col_list, value_name="Delta usage change", var_name="Years", ignore_index=False)
    df_melt = df_melt[df_melt["Delta usage change"] != 0]
    
    plt.figure(figsize=(6, 9))
    sns.barplot(data=df_melt, y=df_melt.index, x="Delta usage change", hue="Years", orient="h")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("")
    plt.title(
        f"Proportion of Keyword by Time Period\n{n_gram}_gram{'; Medical' if med else ''}{'; Abbreviations' if abb else ''}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Plot", "Keywords", f"{n_gram}_gram", abb_med, f"{n_gram}_gram{abb_med}_Delta_Change.png"), dpi=400, transparent=True)
    # plt.show()
    
    plt.close('all')





if __name__ == "__main__":
    import yaml
    os.chdir(os.path.dirname(os.getcwd()))
    from misc.prep_config import prep_config, prep_config_kwargs
    from misc.setup_logger import setup_logger
    
    with open(os.path.join(os.getcwd(), "config.yml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(os.getcwd(), "config_kwargs.yml")) as f:
        config_kwargs = yaml.safe_load(f)
    
    config = prep_config(config=config, config_kwargs=config_kwargs)
    config_kwargs = prep_config_kwargs(config=config, config_kwargs=config_kwargs)
    
    os.environ["NUMEXPR_NUM_THREADS"] = str(config["processors"])
    
    config_kwargs = setup_logger(config_kwargs=config_kwargs)
    logging.info("Config Prepped")
    med = True
    abb = False


    n_gram=1
    #
    # cumulative_keyword_usage(config=config, cutoff=16, med=med, abb=abb,
    #                          fontsize=20, figure_ratio=3/4, figure_scale=1.5, titlesize="large",
    #                          keyword_city_figure_ratio=3 / 4, keyword_city_figure_scale=2,
    #                          keyword_city_fontsize=16,
    #                          keyword_city_titlesize="large", keyword_city_top_x_cities=20)
