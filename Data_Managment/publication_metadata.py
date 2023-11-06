import os, logging
import pandas as pd

from misc.save_files import write_pickle
def publication_metadata(df_publications: pd.DataFrame, config: dict):
    if "affiliation id" in df_publications.columns:
        return df_publications
    df_publications["aff_dict"] = df_publications.apply(lambda x: primary_affiliation(x["authors"]), axis=1)
    df_publications = pd.concat([df_publications.drop(["aff_dict"], axis=1), df_publications["aff_dict"].apply(pd.Series)], axis=1)
    write_pickle(path=os.path.join(config["System"]["base_dir"], "dataset.pkl"), obj=df_publications,
                 logger=logging.getLogger(config["System"]["logging"]["logger_name"]))
    return df_publications


def primary_affiliation(l: list):
    if len(l) == 0:
        return {"institute": None, "country": None,
                 "city": None, "affiliation id": None}
    temp_dict_first = {"institute": l[0].get("institute", None) , "country": l[0].get("country", None),
                 "city": l[0].get("city", None), "affiliation id": l[0].get("affiliation_id", None)}
    if len(l) == 1:
        return temp_dict_first
    temp_dict_last = {"institute": l[-1].get("institute", None) , "country": l[-1].get("country", None),
                 "city": l[-1].get("city", None), "affiliation id": l[-1].get("affiliation_id", None)}
    if len([i for i in temp_dict_first.values() if i is not None]) > len([i for i in temp_dict_last.values() if i is not None]):
        return temp_dict_first
    else:
        return temp_dict_last

