import logging, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from misc.visualisations import set_figure_estetics
from misc.save_files import read_pickle
from misc.save_figures import save_plt_figure


from wordcloud import WordCloud, get_single_color_func

class MyColorFunctor():
    def __init__(self,frequencies):
        self.frequencies = frequencies
        # self.colormap = colormap

    def __call__(self,word,font_size,position,orientation,random_state=None,**kwargs):
        # return f"{self.colormap}({360 * self.frequencies[word]}, 80%%, 50%%)"
        return f"hsl({360 * self.frequencies[word]}, 80%%, 50%%)"

def color_func(*args, **kwargs):
    font_size_range = (5, 60)
    cmap = plt.get_cmap("coolwarm")

    norm = plt.Normalize(vmin=font_size_range[0], vmax=font_size_range[1])
    # print([int(i * 255) for i in cmap(norm(kwargs.get("font_size", font_size_range[0])))])
    return tuple(int(i * 255) for i in cmap(norm(kwargs.get("font_size", font_size_range[0]))))


def full_wordcloud(config: dict, n_gram: int = 1, med: bool = False, abb: bool = False,
                   top_n_words=100,
                   figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                   titlesize: str = "xx-large"):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med,
                               f"{n_gram}_gram{abb_med}_wordcloud_full.png")
    target_data_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                                    f"tfidf_{n_gram}_gram{abb_med}_filterable.pkl")
    if os.path.isfile(target_file):
        return
    set_figure_estetics(figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)

    # <editor-fold desc="Load data">
    df = read_pickle(target_data_file, logger=logger)
    if df is None:
        df = read_pickle(target_data_file, logger=logger)
        if df is None:
            logger.error(
                f"WordCloud Generation - {target_data_file} - No such file exist - potentially no occurances in the dataset")
            return
    # </editor-fold>

    # <editor-fold desc="Generate a wordcloud for a time interval">
    freq = {word: freq for word, freq in zip(df.index, df["sum"]) if freq > 0}
    wc = WordCloud(background_color="white", max_words=top_n_words,
                   color_func=get_single_color_func("black"), #color_func=color_func
                   min_font_size=5, max_font_size=60)
    wc.generate_from_frequencies(frequencies=freq)

    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation="bilinear")
    # </editor-fold>

    # <editor-fold desc="Figure esthetics and saving">
    plt.axis("off")
    plt.tight_layout()
    save_plt_figure(path=target_file, logger=logger, dpi=800)
    # </editor-fold>


def timeinterval_wordcloud(config: dict, n_gram: int = 1, med: bool = False, abb: bool = False,
                           top_n_words=100,
                           figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                           titlesize: str = "xx-large"):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))

    target_folder = os.path.join(config["System"]["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med)
    # <editor-fold desc="Load data">
    target_data_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                                    f"{n_gram}_gram{abb_med}_Keyword_by_Timeinterval.pkl")

    df = read_pickle(target_data_file, logger=logger)
    if df is None:
        df = read_pickle(target_data_file, logger=logger)
        if df is None:
            logging.error(
                f"Word cloud generation - {n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv - No such file exist - potentially no occurances in the dataset")
            return
    # </editor-fold>

    years = sorted(df.columns)

    # <editor-fold desc="select appropriate colors">
    if len(years) > len(config["Visualisations"]["categorial_colorfull_colors"]):
        palette = sns.color_palette(config["Visualisations"]["diverging_colors"], desat=1, n_colors=len(years))
    else:
        palette = config["Visualisations"]["categorial_colorfull_colors"][:len(years)]
    # </editor-fold>

    set_figure_estetics(figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)

    for y, color in zip(years, palette):
        # <editor-fold desc="Skip Already Generated Figures and empty wordclouds">
        if df[df[y] > 0].shape[0] == 0:
            continue
        target_file = os.path.join(target_folder, f"{n_gram}_gram{abb_med}_wordcloud_{y}.png")
        if os.path.isfile(target_file):
            continue
        # </editor-fold>

        # <editor-fold desc="Generate a wordcloud for a time interval">
        wc = WordCloud(background_color="white", max_words=top_n_words)
        wc.generate_from_frequencies(frequencies={word: freq for word, freq in zip(df.index, df[y]) if freq > 0})
        plt.figure(figsize=(20,10))
        plt.imshow(wc.recolor(colormap=sns.dark_palette(color=color, as_cmap=True)), interpolation="bilinear")
        # </editor-fold>

        # <editor-fold desc="Figure esthetics and saving">
        plt.axis("off")
        plt.tight_layout()
        save_plt_figure(path=target_file, logger=logger, dpi=800)
        # </editor-fold>


def generate_wordcloud(config: dict, n_gram: int = 1, med: bool = False, abb: bool = False,
                       top_n_words=100,
                       figure_ratio: float = 3 / 4, figure_scale: float = 1, fontsize: int = 10,
                       titlesize: str = "xx-large"):
    full_wordcloud(config=config, n_gram=n_gram, med=med, abb=abb,
                   top_n_words=top_n_words,
                   figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize,
                   titlesize=titlesize)
    timeinterval_wordcloud(config=config, n_gram=n_gram, med=med, abb=abb,
                           top_n_words=top_n_words,
                           figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize,
                           titlesize=titlesize)

