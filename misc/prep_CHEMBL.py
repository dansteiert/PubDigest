import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import enchant


def prep_CHEMBL():
    Chembl_file = os.path.join(os.getcwd(), "base_data", "CHEMBL_medication_names.csv")
    with open(Chembl_file, "r") as f:
        if not ";" in f.readline():
            print("File already processed!")
            return
    df = pd.read_csv(Chembl_file, index_col=[0], sep=";")

    meds = [*[term.lower()
            for syn_string in df["Synonyms"].tolist()
            if not isinstance(syn_string, float)
            for term in syn_string.split("|")],
            *[term.lower()
            for term in df["Name"].tolist()
            if not isinstance(term, float)]]

    d = enchant.Dict("en_US")
    misspelled_meds = list({term for term in meds if not d.check(term)})

    pd.DataFrame(data={"Synonyms": misspelled_meds}).to_csv(Chembl_file)




if __name__ == "__main__":
    prep_CHEMBL()