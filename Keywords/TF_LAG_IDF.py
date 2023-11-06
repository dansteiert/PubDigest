import os
import sys
sys.path.append(os.getcwd())
from gensim import corpora
from gensim.models import TfidfModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy import stats
from matplotlib.colors import LogNorm
import re

def tf_lag_idf(lemma_list, year_list, target_path, medication=False, abbreviations=False):
    
    # dfs: document frequency- how many documents contain this keyword
    # cfs: collection frequency - how many instances of the element are within the collection
    year_dict = {}
    for y, l in zip(year_list, lemma_list):
        temp_list = year_dict.get(y, [])
        temp_list.append(l)
        year_dict[y] = temp_list
    year_set = sorted(set(year_list))
    year_dictionary = corpora.Dictionary(lemma_list)
    if medication and abbreviations:
        df_med = pd.read_csv(os.path.join(os.getcwd(), "base_data", "CHEMBL_medication_names.csv"))
        good_ids = [year_dictionary.token2id.get(i, None) for i in df_med["Synonyms"]]
        good_ids = [i for i in good_ids if i is not None]
        good_ids = [*good_ids, *[v for k, v in year_dictionary.token2id.items() if bool(re.match(r'\w*[A-Z]\w*', k))]]
        year_dictionary.filter_tokens(good_ids=good_ids)

    elif medication:
        df_med = pd.read_csv(os.path.join(os.getcwd(), "base_data", "CHEMBL_medication_names.csv"))
        good_ids = [year_dictionary.token2id.get(i, None)for i in df_med["Synonyms"]]
        good_ids = [i for i in good_ids if i is not None]
        year_dictionary.filter_tokens(good_ids=good_ids)
    elif abbreviations:
        good_ids = [v for k, v in year_dictionary.token2id.items() if bool(re.match(r'\w*[A-Z]\w*', k))]
        year_dictionary.filter_tokens(good_ids=good_ids)

    else:
        year_dictionary.filter_extremes(no_below=2)
    if len(year_dictionary.token2id) == 0:
        return None

    # max_docs = max([len(v) for v in year_dict.values()])
    # empty_lists = [[]] * (max_docs)
    # year_dict = {k: [*v, *empty_lists[:(max_docs - len(v))]] for k, v in year_dict.items()}


    year_lag_idf_dict = {}
    year_idf_dict = {}
    year_tf_dict = {}

    for index_y, y in enumerate(year_set):
        year_lemma_list = year_dict.get(y, None)
        year_BoW_corpus = [year_dictionary.doc2bow(doc) for doc in year_lemma_list]
        year_tfidf_model = TfidfModel(corpus=year_BoW_corpus, smartirs='ntc')


        # <editor-fold desc="Calculate the Term frequency by year">
        temp_dict = {}
        total_terms = 0
        for doc in year_tfidf_model[year_BoW_corpus]:
            total_terms += sum([freq for id, freq in doc])
            # retrieve original names of the terms and their frequencies
            doc_dict = {year_dictionary.get(id, None): freq for id, freq in doc}
            
            # sum up the frequency of all same terms
            temp_dict = {k: temp_dict.get(k, 0) + doc_dict.get(k, 0) for k in
                         set([*temp_dict.keys(), *doc_dict.keys()])}
        # get term frequency by year
        year_tf_dict[y] = {k: v/total_terms for k, v in temp_dict.items()}
        # </editor-fold>

        # <editor-fold desc="Calculate the Lag Inverse Document frequency by year">
        if index_y + 1 < len(year_set):
            # get IDFs from the model
            idf_dict = year_tfidf_model.idfs
            
            # convert ids into terms
            year_lag_idf_dict[year_set[index_y + 1]] = {year_dictionary.get(id, None): idf for id, idf in idf_dict.items()}
        year_idf_dict[y] = {year_dictionary.get(id, None): idf for id, idf in idf_dict.items()}
        # </editor-fold>


    df_lag_idf = pd.DataFrame(data=year_lag_idf_dict)
    df_lag_idf = df_lag_idf.fillna(max(df_lag_idf.max()))
    
    df_idf = pd.DataFrame(data=year_idf_dict)
    max_idf = max(df_idf.max())
    df_idf = df_idf * (-1)
    df_idf = df_idf + max_idf
    df_idf = df_idf.fillna(0)

   
    df_tf = pd.DataFrame(data=year_tf_dict)
    df_tf = df_tf.fillna(0)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # df_tf = pd.DataFrame(min_max_scaler.fit_transform(df_tf.T).T, columns=df_tf.columns, index=df_tf.index)

    
    if df_tf.shape[0] == 0:
        return None
    
    df_tf_lag_idf = df_tf * df_lag_idf
    df_tf_lag_idf = df_tf_lag_idf.fillna(0)
    df_tf_lag_idf.to_csv(target_path)

    
    
    # df_melted = df_tf_lag_idf.reset_index(drop=False).melt(id_vars=["index"], value_name="TFLAGIDF",
    #                                            var_name="years").dropna().sort_values(by="TFLAGIDF", ascending=False).head(50)
    # sns.barplot(data=df_melted, y="index", x="TFLAGIDF", hue="years")
    # plt.tight_layout()
    # plt.savefig(target_path[:-4] + "_top50.png", dpi=400)
    # plt.close("all")
    
    
    # df_tf_lag_idf = df_tf_lag_idf[(np.abs(stats.zscore(df_tf_lag_idf)) < 3).any(axis=1)]
    # min_max_scaler = preprocessing.MinMaxScaler()
    # df_tf_lag_idf = pd.DataFrame(min_max_scaler.fit_transform(df_tf_lag_idf.T).T, columns=df_tf_lag_idf.columns, index=df_tf_lag_idf.index)
    # df_tf_lag_idf = df_tf_lag_idf.drop(sorted(df_tf_lag_idf.columns)[0], axis=1)

    df_tf_lag_idf = df_tf_lag_idf.sort_values(by=sorted(df_tf_lag_idf.columns))
    sns.heatmap(df_tf_lag_idf, cmap=sns.color_palette("Blues", as_cmap=True), norm=LogNorm())
    plt.title("TF-LAG-IDF Values by Year")
    plt.tight_layout()
    plt.savefig(target_path[:-4] + ".png", dpi=400)
    plt.close("all")
    
    df_idf = df_idf.sort_values(by=sorted(df_idf.columns))
    # print(df_idf.describe())
    sns.heatmap(df_idf, cmap=sns.color_palette("Blues", as_cmap=True))
    plt.title("IDF Values by Year")
    plt.tight_layout()
    plt.savefig(target_path[:-4] + "_IDF_only.png", dpi=400)
    plt.close("all")

    # for col_name in df_tf_lag_idf.columns:
    #     print((np.count_nonzero(df_tf_lag_idf[col_name])), col_name)
    # try:
    #     df_tf_lag_idf[col_name] / (1 / np.count_nonzero(df_tf_lag_idf[col_name]))
    # except:
    #     pass
    #
    # idx = msk.all(axis=1)
    # df[idx]