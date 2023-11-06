import os, logging
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from misc.visualisations import set_figure_estetics
from misc.save_figures import save_plt_figure
from misc.save_files import read_pickle

def plot_associated_disease_names(config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    origin_file = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", "disease_name_counts.pkl")
    df_disease_count = read_pickle(path=origin_file, logger=logger)

    df_disease_count = df_disease_count.sort_values(by=["term count"], ascending=False).head(20)
    target_file = os.path.join(config["System"]["base_dir"], "Plot", "Disease_names.png")

    set_figure_estetics(y_axis_elements=df_disease_count.shape[0], figure_ratio=10 / 9,
                        figure_scale=3, fontsize=16, titlesize="large")
    sns.barplot(data=df_disease_count, y="term count", x="disease name", color=config["Visualisations"]["single_color"])
    # sns.barplot(data=df_disease_count, y="term count", x="disease name", palette=reversed(sns.color_palette(config["Visualisations"]["diverging_colors"], n_colors=df_disease_count.shape[0])))
    if config["Visualisations"]["with_title"]:
        plt.title("Disease names counts")
    plt.xlabel("")
    plt.ylabel("")
    plt.yticks(ticks=np.arange(0, df_disease_count["term count"].max() + 10, 25))
    plt.xticks(rotation=90)
    plt.grid(axis="y")
    plt.tight_layout()
    save_plt_figure(path=target_file, logger=logger)
