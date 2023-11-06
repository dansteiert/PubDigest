import os, logging
import pandas as pd

from misc.save_files import read_pickle, write_pickle,read_pandas_from_csv
from misc.abb_med_disease import gen_abb_med_disease



def generate_keyword_plotting_files(config: dict, df_publications: pd.DataFrame):
    n_gram_list = [index for index in config["NLP"]["n_gram_list"]
                   if not os.path.isfile(os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords",
                                                      f"{index}_gram_Keyword_by_Timeinterval.csv"))]

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    if len(n_gram_list) == 0:
        logger.info("n_gram_list is empty -  data keywords by timeinterval")
        return


    # Define time intervals and split years into respective groups
    df_timeinterval = read_pickle(os.path.join(config["System"]["base_dir"], "Data_for_Plotting",
                                               "Publications_by_TimeIntervals.pkl"), logger=logger)
    if df_timeinterval is None:
        df_timeinterval = get_time_intervals(df=df_publications, config=config)



    # <editor-fold desc="group the tfidf values and save results">
    disease = False
    for i in n_gram_list:
        for med in [True, False]:
            for abb in [True, False]:
                # <editor-fold desc="Load data">
                abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

                target_directory = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", f"{i}")
                keyword_file_generation_wrapper(target_directory=target_directory, abb_med=abb_med, i=i, logger=logger,
                                                df_timeinterval=df_timeinterval, filter_criterion=config["NLP"]["tfidf_filter_criterion"])
        disease = True
        abb_med = gen_abb_med_disease(abb=abb, med=med, disease=disease)

        target_directory = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Keywords", f"{i}")
        keyword_file_generation_wrapper(target_directory=target_directory, abb_med=abb_med, i=i, logger=logger,
                                        df_timeinterval=df_timeinterval,
                                        filter_criterion=config["NLP"]["tfidf_filter_criterion"])
    # </editor-fold>


def keyword_file_generation_wrapper(target_directory: str, abb_med: str, i: int, logger: logging.Logger,
                                    df_timeinterval: pd.DataFrame, filter_criterion: str):
    df_tfidf = read_pickle(os.path.join(target_directory,
                                        f"tfidf_{i}_gram{abb_med}.pkl"), logger=logger)
    if df_tfidf is None:
        logger.error(f"tfidf_{i}_gram{abb_med}.csv - File does not exist since there are no entries")
        return
    # </editor-fold>

    # <editor-fold desc="Sort by overall tfidf and calculate tfidf for each time period">
    df_tfidf = sort_keywords_by_tfidf(df_tfidf=df_tfidf, df_timeinterval=df_timeinterval)
    df_intervals = split_keywords_in_timeintervals(df_tfidf=df_tfidf, df_timeinterval=df_timeinterval,
                                                   target_file=os.path.join(target_directory,
                                                                            f"{i}_gram{abb_med}_Keyword_by_Timeinterval.pkl"),
                                                   logger=logger)
    # </editor-fold>

    # <editor-fold desc="Calculate TFIDF and add column to filter by e.g. 'avg'">
    filter_for_most_important(df_tfidf=df_tfidf, df_timeinterval=df_timeinterval,
                              by=filter_criterion,
                              target_file=os.path.join(target_directory,
                                                       f"tfidf_{i}_gram{abb_med}_filterable.pkl"),
                              logger=logger)

    # </editor-fold>

    # <editor-fold desc="Generate Data for the Mosaik Plot">
    data_keywords_as_mosaik(df_intervals,
                            target_file=os.path.join(target_directory,
                                                     f"tfidf_{i}_gram{abb_med}_Keyword_as_Mosaik.pkl"),
                            logger=logger)

    # </editor-fold>

    # <editor-fold desc="Generate Data for the Venn-like Diagramm">
    data_keywords_venn_scaling(df_intervals,
                               target_file=os.path.join(target_directory,
                                                        f"tfidf_{i}_gram{abb_med}_Keyword_as_Venn_Diagram.pkl"),
                               logger=logger)

    # </editor-fold>


def sort_keywords_by_tfidf(df_tfidf: pd.DataFrame, df_timeinterval: pd.DataFrame):
    """
    
    :param df_tfidf:
    :param df_timeinterval:
    :return: sum describes the global tf-idf value
    """
    total_documents = df_timeinterval["count"].sum()
    df_tfidf["sum"] = df_tfidf[[i for i in df_tfidf.columns if i != "idf"]].sum(axis=1)
    df_tfidf["sum"] = df_tfidf["sum"] / total_documents
    df_tfidf["sum"] = df_tfidf["sum"] * df_tfidf["idf"]
    df_tfidf = df_tfidf.sort_values(by="sum", ascending=False)
    return df_tfidf


def split_keywords_in_timeintervals(df_tfidf: pd.DataFrame, df_timeinterval: pd.DataFrame, target_file: str, logger: logging.Logger):
    if os.path.isfile(target_file):
        return read_pickle(path=target_file, logger=logger)
    df_tfidf_by_interval = pd.DataFrame()
    corpus_docs = df_timeinterval["count"].sum()

    for interval in df_timeinterval["Time Interval"].tolist():
        interval_years = df_timeinterval[df_timeinterval["Time Interval"] == interval]["publication year"].tolist()
        interval_years = [int(float(i)) for i in interval_years]
        interval_docs = df_timeinterval[df_timeinterval["Time Interval"] == interval]["count"].sum()
        df_tfidf_by_interval[interval] = (df_tfidf[interval_years].sum(axis=1) * df_tfidf["idf"]) * corpus_docs/interval_docs
    write_pickle(target_file, obj=df_tfidf_by_interval, logger=logger)
    return df_tfidf_by_interval

def gen_interval_dict(df_timeinterval: pd.DataFrame):
    interval_dict = {}
    for i in range(0, int(df_timeinterval["Time Interval"].max()) + 1):
        dates = df_timeinterval[df_timeinterval["Time Interval"] == i]["publication year"].tolist()
        if min(dates) == max(dates):
            interval_dict[i] = f"{min(dates)}"
        else:
            interval_dict[i] = f"{min(dates)}-{max(dates)}"
    return interval_dict


def data_keywords_as_mosaik(df: pd.DataFrame, target_file: str, logger: logging.Logger):
    if os.path.isfile(target_file):
        return
    df_copy = df.copy()
    column_list = sorted(df_copy.columns)
    added = []
    for index, i in enumerate(column_list):
        if index + 1 >= len(column_list):
            continue
        added.append(i)
        df_copy["{} bottom".format(column_list[index + 1])] = df_copy.loc[:, added].sum(axis=1)
    write_pickle(path=target_file, obj=df_copy, logger=logger)


def data_keywords_venn_scaling(df: pd.DataFrame, target_file: str, logger: logging.Logger):
    if os.path.isfile(target_file):
        return
    df_copy = df.copy()
    column_list = sorted(df_copy.columns)
    df_copy["100%"] = df.loc[:, column_list].sum(axis=1)
    for i in column_list:
        df_copy[i] = df_copy[i]/df_copy["100%"]
    df_copy = df_copy.drop(["100%"], axis=1)
    added = []
    for index, i in enumerate(column_list):
        if index + 1 >= len(column_list):
            continue
        added.append(i)
        df_copy["{} bottom".format(column_list[index + 1])] = df_copy.loc[:, added].sum(axis=1)
    write_pickle(path=target_file, obj=df_copy, logger=logger)


def get_time_intervals(df: pd.DataFrame, config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    target_path = os.path.join(config["System"]["base_dir"], "Data_for_Plotting", "Publications_by_TimeIntervals.pkl")
    if os.path.isfile(target_path):
        return read_pickle(path=target_path, logger=logger)

    # <editor-fold desc="Load data">
    if df is None:
        df = read_pickle(os.path.join(config["System"]["base_dir"], "dataset.pkl"))
    # </editor-fold>

    df["publication year"] = df["publication year"].astype(int)


    # <editor-fold desc="Determine Time Intervals">
    # "Fixed" Timeintervals 1, 2, 2, 4, 4, 8, 8, 16, 16, ...
    min_year = int(df["publication year"].min())
    max_year = int(df["publication year"].max())
    max_per_interval = config["Query"]["starting_interval_size"]
    counter = config["Query"]["starting_counter"]
    interval = 0
    interval_counter = 0
    group_dict = {}

    for i in range(int(max_year), int(min_year) - 1, -1):
        group_dict[i] = interval
        counter += 1
        if counter == max_per_interval:
            counter = 0
            interval_counter += 1
            if interval_counter == 2:
                interval_counter = 0
                max_per_interval *= 2
            interval += 1
    # </editor-fold>


    # <editor-fold desc="Save Publication dates, with count and groups">
    df_groupby = df[["publication year", "PMID"]].groupby(by="publication year").count()

    df_groupby = df_groupby.rename(columns={"PMID": "count"})
    df_groupby = df_groupby.reset_index(drop=False)
    df_groupby = df_groupby.astype({"publication year": int})


    df_groupby["Time Interval"] = df_groupby.apply(lambda x: group_dict.get(int(x["publication year"])), axis=1)
    pub_date = [int(i) for i in df_groupby["publication year"].tolist()]
    years = [int(y) for y in range(min_year, max_year + 1) if y not in pub_date]
    if len(years) > 0:
        df_groupby = df_groupby.append(pd.DataFrame(data={"publication year": years, "count": [0] * len(years)}), ignore_index=True)
    df_groupby = df_groupby.sort_values(by=["publication year"])
    df_groupby["Time Interval"] = df_groupby["Time Interval"].fillna(method="ffill")



    interval_dict = gen_interval_dict(df_timeinterval=df_groupby)
    df_groupby["Time Interval"] = df_groupby.apply(lambda x: interval_dict.get(x["Time Interval"], "None"), axis=1)


    write_pickle(path=target_path, obj=df_groupby, logger=logger)
    # </editor-fold>
    return df_groupby


def filter_for_most_important(df_tfidf: pd.DataFrame, df_timeinterval: pd.DataFrame, target_file: str,
                              logger: logging.Logger, by: str = "avg"):
    if os.path.isfile(target_file):
        return
    column_list = sorted([i for i in df_tfidf.columns if i not in ["idf", "sum"]])
    for i in column_list:
        # tf per year or time interval * ((publications count/total publications) * global idf)
        df_tfidf[i] = df_tfidf[i] * ((df_timeinterval[df_timeinterval["publication year"] == int(i)]["count"].min()/df_timeinterval["count"].sum()) * df_tfidf["idf"])
    df_tfidf = df_tfidf.fillna(0)
    if by == "avg":
        df_tfidf["avg"] = df_tfidf.apply(lambda x: avg_row_value(row=x, column_list=column_list), axis=1)
        write_pickle(path=target_file, obj=df_tfidf, logger=logger)
    else:
        logger.error("Filter criterion is not implemented! -> Currently implemented are: 'avg'")


def avg_row_value(row, column_list):
    for enu, i in enumerate(column_list):
        if row[i] != 0:
            return row.loc[column_list[enu:]].mean()


if __name__ == "__main__":
    os.chdir(os.path.split(os.getcwd())[0])
    # search_term = "Chronic thromboembolic pulmonary hypertension"
    # search_term = "(Pulmonary Arterial Hypertension) AND (Proliferation)"
    search_term = "((((Serine) OR (Threonine)) OR (Tyrosine)) AND (Kinase)) AND (Pulmonary Arterial Hypertension)"
    # search_term = "pharmacophore"
    # base_dir = os.path.join(os.path.split(os.getcwd())[0], "data", search_term)
    base_dir = os.path.join(os.getcwd(), "data", search_term)
    data_keywords_by_timeinterval(base_dir=base_dir, ngram_ns=[1])
    # get_time_intervals(base_dir=base_dir)