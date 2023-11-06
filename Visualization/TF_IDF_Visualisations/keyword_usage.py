import logging, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from misc.visualisations import set_figure_estetics
from misc.save_files import read_pickle, read_pandas_from_csv
from misc.counter import count_dict

from Data_Managment.build_DocTermDict import DocumentTermDictionary


from Keywords.Time_intervals import generate_keyword_plotting_files

# TODO: entire File!
def keyword_usages_warpper(config: dict, query_terms: list, df: pd.DataFrame = None,
                           df_cities: pd.DataFrame = None, top_x_cities: int = 40, top_x_countries: int = 5,

                           figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                           abb_med: str = None, n_gram: int = None, titlesize="xx-large"):
    # <editor-fold desc="Load data">
    if df is None:
        df = read_pickle(path=os.path.join(config["base_dir"], "dataset.pkl"))

    if df_cities is None:
        df_cities = read_pickle(path=os.path.join(config["base_dir"], "Institutes.pkl"))

    df["is_research_article"] = df.apply(lambda x: True if x["iCite_is_research_article"] == "Yes" else False, axis=1)
    df["is_clinical"] = df.apply(lambda x: True if x["iCite_is_clinical"] == "Yes" else False, axis=1)
    df.loc[:, "article_type"] = df.apply(
        lambda x: "Clinical Research" if x["is_research_article"] and x["is_clinical"] else (
            "Research" if x["is_research_article"] and not x["is_clinical"] else (
                "Clinical Trial" if not x["is_research_article"] and x["is_clinical"] else "Other")), axis=1)

    df = df[["PMID", "main affiliation", "publication year", "article_type"]]
    df_cities = df_cities[["country", "city"]]
    df = pd.merge(left=df, right=df_cities, how="left", left_on="main affiliation", right_index=True)

    term_dict = DocumentTermDictionary()
    term_dict = term_dict.load(path=os.path.join(config["base_dir"], "Term_dictionary.dict"))

    # </editor-fold>
    if query_terms is None:
        df_tfidf = read_pandas_from_csv(
            os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                         f"tfidf_{n_gram}_gram{abb_med}_filterable.csv"))
        if df_tfidf is None:
            generate_keyword_plotting_files(config=config)
            df = read_pandas_from_csv(
                os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                             f"tfidf_{n_gram}_gram{abb_med}_filterable.csv"))
            if df_tfidf is None:
                logging.error(
                    f"Word cloud generation - {n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv - No such file exist - potentially no occurances in the dataset")
                return
        df_tfidf = df_tfidf.sort_values(by=["sum"], ascending=False)
        query_terms = df_tfidf.index.tolist()[:6]

    [plot_keyword_usage_single(config=config, query_term=i, term_dict=term_dict, df=df, top_x_cities=top_x_cities,
                               top_x_countries=top_x_countries,
                               figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, abb_med=abb_med,
                               n_gram=n_gram, titlesize=titlesize) for i in query_terms]
    plot_keyword_usage_group(config=config, query_terms=query_terms, term_dict=term_dict, df=df,
                             top_x_cities=top_x_cities,
                             top_x_countries=top_x_countries,
                             figure_ratio=figure_ratio, figure_scale=figure_scale, fontsize=fontsize, abb_med=abb_med,
                             n_gram=n_gram, titlesize=titlesize)


def plot_keyword_usage_single(config: dict, query_term: str, term_dict: DocumentTermDictionary,
                              df: pd.DataFrame,
                              top_x_cities: int = 40, top_x_countries: int = 5,
                              figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                              abb_med: str = None, n_gram: int = None, titlesize="xx-large"):
    target_file = os.path.join(config["base_dir"], "Plot", "Keywords", "Keyword_usage", f"{n_gram}_gram", abb_med,
                               f"keyword_usage_{query_term}.png")
    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    set_figure_estetics(y_axis_elements=top_x_cities, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>
    # <editor-fold desc="get documents in which the keyword appeared">
    key_doc = term_dict.term2key_doc.get(query_term, None)
    docs = key_doc[1]
    # </editor-fold>

    # <editor-fold desc="filter for those documents and join with on the location id">
    df_temp = df[df.index.isin(docs)]
    df_temp = df_temp.dropna()
    # </editor-fold>

    # <editor-fold desc="convert dates to datetime objects">
    df_temp = df_temp.sort_values(by=["publication year"])
    df_temp = df_temp[df_temp["article_type"].isin(set(["Clinical Research", "Research"]))]

    # </editor-fold>

    # <editor-fold desc="select the first cities publishing something with the query term + the names of the first countries, name the other ones 'other'">
    city_names = df_temp["city"].unique()[:top_x_cities]
    df_temp = df_temp[df_temp["city"].isin(city_names)]
    country_names = df_temp.groupby(by=["country"])["city"].count().sort_values(ascending=False).head(
        top_x_countries).index.tolist()
    df_temp["country"] = df_temp.apply(lambda x: x["country"] if x["country"] in country_names else "other", axis=1)
    # </editor-fold>

    # <editor-fold desc="select appropriate colors">
    if df_temp["country"].unique().shape[0] > len(config["categorial_colorfull_colors"]) + 1:
        palette = sns.color_palette(palette=config["diverging_colors"], desat=1,
                                    n_colors=df_temp["country"].unique().shape[0])
    else:
        palette = config["categorial_colorfull_colors"][:df_temp["country"].unique().shape[0]]
    # </editor-fold>

    # <editor-fold desc="Plot scatterplot of cities publishing the query term">

    hue_order = [*country_names, "other"] if "other" in list(df_temp["country"].unique()) else country_names
    sns.scatterplot(data=df_temp, x="publication year", y="city", hue="country",
                    hue_order=hue_order, style="article_type",
                    style_order=["Clinical Research", "Research"],
                    palette=palette)

    # </editor-fold>
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="medium", framealpha=0.2)
    # plt.legend(loc="lower left", fontsize="medium", framealpha=0.2)
    # plt.legend(loc="lower left", fontsize="small", framealpha=0.2)
    # <editor-fold desc="Figure esthetics and saving">
    plt.xticks(rotation=90)
    plt.xlabel("Year")
    plt.title(f"Time resolution for the usage of\n{query_term}")
    plt.tight_layout()
    plt.savefig(target_file, dpi=400, transparent=True)
    plt.close("all")
    # </editor-fold>


def plot_keyword_usage_group(config: dict, query_terms: list, term_dict: DocumentTermDictionary,
                             df: pd.DataFrame,
                             top_x_cities: int = 40, top_x_countries: int = 5,
                             figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                             abb_med: str = None, n_gram: int = None, titlesize="xx-large"):
    target_file = os.path.join(config["base_dir"], "Plot", "Keywords", "Keyword_usage", f"{n_gram}_gram", abb_med,
                               f"keyword_usage_top_keywords.png")
    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    set_figure_estetics(y_axis_elements=top_x_cities, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    key_doc_dict = {i: term_dict.term2key_doc.get(i, [])[1] for i in query_terms}

    # <editor-fold desc="filter for those documents and join with on the location id">
    df_temp = df[df.index.isin(set([j for i in key_doc_dict.values() for j in i]))]
    for k, v in key_doc_dict.items():
        df_temp[k] = df_temp.apply(lambda x: True if x.name in set(v) else False, axis=1)
    df_temp = df_temp.melt(id_vars=["PMID", "main affiliation", "publication year", "article_type", "country", "city"],
                           value_vars=key_doc_dict.keys(), value_name="Drug_occurance", var_name="Drug")
    df_temp = df_temp[df_temp["Drug_occurance"]]
    df_temp = df_temp.dropna()
    # </editor-fold>

    # <editor-fold desc="convert dates to datetime objects">
    df_temp["publication year"] = df_temp["publication year"].astype(int)

    df_temp = df_temp.sort_values(by=["publication year"])
    df_temp = df_temp[df_temp["article_type"].isin(set(["Clinical Research", "Research"]))]

    # </editor-fold>

    # <editor-fold desc="select the first cities publishing something with the query term + the names of the first countries, name the other ones 'other'">
    country_names = df_temp.groupby(by=["country"])["city"].count().sort_values(ascending=False).head(
        top_x_countries).index.tolist()
    df_temp["country"] = df_temp.apply(lambda x: x["country"] if x["country"] in country_names else "other", axis=1)
    city_name_dict = {i: df_temp[df_temp["Drug"] == i]["city"].unique()[:top_x_cities]
                      for i in query_terms}
    mask_list = []
    for k, v in city_name_dict.items():
        mask_list.append((df_temp["Drug"] == k) & (df_temp["city"].isin(v)))
    mask = pd.DataFrame(mask_list).transpose().any(axis=1)
    df_temp["city_mask"] = mask
    df_temp["city"] = df_temp.apply(lambda x: x["city"] if x["city_mask"] else "other", axis=1)
    # </editor-fold>
    hue_order = [*country_names, "other"] if "other" in list(df_temp["country"].unique()) else country_names

    # <editor-fold desc="select appropriate colors">
    if len(hue_order) > len(config["categorial_colorfull_colors"]) + 1:
        palette = sns.color_palette(palette=config["diverging_colors"], desat=1,
                                    n_colors=len(hue_order))
    else:
        palette = config["categorial_colorfull_colors"][:len(hue_order)]
    # </editor-fold>

    # <editor-fold desc="Plot scatterplot of cities publishing the query term">
    g = sns.relplot(data=df_temp, x="publication year", y="city", hue="country", col="Drug", col_wrap=2,
                    col_order=query_terms,
                    kind="scatter", facet_kws={'sharey': False, 'sharex': True},
                    hue_order=hue_order,
                    style="article_type",
                    style_order=["Clinical Research", "Research"],
                    palette=palette, s=120, legend="brief"
                    )

    g.set(xticks=np.arange(df_temp["publication year"].min(), df_temp["publication year"].max(), 3))
    g.map(plt.grid, axis="x", color=config["single_color"])
    g.map(plt.xticks, rotation=90, color="#000000")
    plt.suptitle(f"Time resolution for the usage of\n{', '.join(query_terms)}")
    plt.legend(loc='center left', bbox_to_anchor=(2.5, 0.5))
    for lh in g._legend.legendHandles:
        lh._sizes = [120]

    plt.tight_layout()
    # plt.show()
    plt.savefig(target_file, dpi=400, transparent=True)
    plt.close("all")
    # </editor-fold>


def cumulative_keyword_usage(config: dict, cutoff: int = 40, n_gram: int = 1,
                             med: bool = False, abb: bool = False,
                             figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                             titlesize: str = "xx-large", keyword_city_titlesize: str = "xx-large",
                             keyword_city_figure_ratio: float = 4 / 3, keyword_city_figure_scale: float = 1,
                             keyword_city_fontsize: int = 10, keyword_city_top_x_cities: int = 20):
    # <editor-fold desc="Set Figure sizes and respective labels">
    abb_med = '_abb_med' if med and abb else ('_med' if med else ('_abb' if abb else ''))
    os.makedirs(os.path.join(config["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med),
                exist_ok=True)

    set_figure_estetics(figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize, titlesize=titlesize)
    # </editor-fold>

    # <editor-fold desc="Load data">

    df = read_pandas_from_csv(
        os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                     f"tfidf_{n_gram}_gram{abb_med}_filterable.csv"))
    if df is None:
        generate_keyword_plotting_files(config=config)
        df = read_pandas_from_csv(
            os.path.join(config["base_dir"], "Data_for_Plotting", "Keywords", f"{n_gram}",
                         f"tfidf_{n_gram}_gram{abb_med}_filterable.csv"))
        if df is None:
            logging.error(
                f"Word cloud generation - {n_gram}_gram{abb_med}_Keyword_by_Timeinterval.csv - No such file exist - potentially no occurances in the dataset")
            return
    # </editor-fold>

    # <editor-fold desc="preprocess">
    df = df.sort_values(by=[config["tfidf_filter_criterion"]], ascending=False).head(cutoff)
    term_dict = DocumentTermDictionary()
    term_dict = term_dict.load(path=os.path.join(config["base_dir"], "Term_dictionary.dict"))
    # </editor-fold>

    result_dict = {}
    for i in df.index.tolist():
        # <editor-fold desc="get documents in which the keyword appeared">
        key_doc = term_dict.term2key_doc.get(i, None)
        # </editor-fold>
        doc_years = [term_dict.doc2year.get(j) for j in key_doc[1]]
        doc_years_dict = count_dict(doc_years)
        result_dict[i] = doc_years_dict
    df_results = pd.DataFrame(data=result_dict)
    df_results = df_results.fillna(0)
    df_results = df_results.sort_index()
    df_results = df_results.cumsum()
    df_results = df_results.reset_index()
    df_melt = df_results.melt(id_vars="index", value_vars=[i for i in df_results.columns if i != "index"],
                              var_name="Keywords")
    df_melt = df_melt.sort_values(by="value", ascending=False)

    # <editor-fold desc="select appropriate colors">
    if df_melt["Keywords"].unique().shape[0] > len(config["categorial_colorfull_colors"]):
        palette = sns.color_palette(palette=config["diverging_colors"], desat=1,
                                    n_colors=df_melt["Keywords"].unique().shape[0])

    else:
        palette = config["categorial_colorfull_colors"][:df_melt["Keywords"].unique().shape[0]]
    # </editor-fold>

    sns.lineplot(data=df_melt, x="index", y="value", hue="Keywords", palette=palette)
    # <editor-fold desc="Figure Estetics">
    # plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="large")
    leg = plt.legend(loc="upper left", ncol=2, framealpha=0.2)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.xlabel("Year")
    plt.ylabel("Cumulative Keyword usage")
    plt.title(
        f"Keyword Usage - Cumulative\n{n_gram}_gram{'; Medical' if med else ''}{'; Abbreviations' if abb else ''}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(config["base_dir"], "Plot", "Keywords", f"{n_gram}_gram", abb_med,
                             f"cumulative_keyword_usage_{n_gram}_gram{abb_med}.png"),
                dpi=400, transparent=True)
    plt.close("all")
    # </editor-fold>

    # keyword_usages_warpper(config=config, query_terms=df_melt["Keywords"], top_x_cities=keyword_city_top_x_cities,
    #                fontsize=keyword_city_fontsize, figure_ratio=keyword_city_figure_ratio,
    #                figure_scale=keyword_city_figure_scale, abb_med=abb_med, n_gram=n_gram, titlesize=keyword_city_titlesize)
