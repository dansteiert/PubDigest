import pandas as pd


def citation_analysis(df_publications: pd.DataFrame, df_authors: pd.DataFrame, config: dict):

    df_temp = df_publications[~df_publications["iCite_is_research_article"].isna()]
    df_temp = df_temp[df_temp["iCite_is_research_article"].apply(lambda x: True if x.lower()=="yes" else False)]
    citation_helper_dict = dict(zip(df_temp.index, df_temp["iCite_citation_count"]))

    df_authors["citation count"] = df_authors.apply(lambda x: citation_aggregation(pmid_list=x["pmid"],
                                                                                   citation_helper_dict=citation_helper_dict), axis=1)
    df_authors["citation count - first author"] = df_authors.apply(lambda x: citation_aggregation(pmid_list=x["pmid_first_author"],
                                                                                   citation_helper_dict=citation_helper_dict), axis=1)
    df_authors["citation count - last author"] = df_authors.apply(lambda x: citation_aggregation(pmid_list=x["pmid_last_author"],
                                                                                   citation_helper_dict=citation_helper_dict), axis=1)
    return df_authors

    # TODO: Implement multiple scores, like H-index, ...

def citation_aggregation(pmid_list: list, citation_helper_dict: dict):
    if not isinstance(pmid_list, list):
        return 0
    return sum([citation_helper_dict.get(i, 0) for i in pmid_list])
