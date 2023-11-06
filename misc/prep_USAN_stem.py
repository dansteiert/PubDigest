import os
import sys
sys.path.append(os.getcwd())
import pandas as pd

def prep_usan():
    usan_file = os.path.join(os.getcwd(), "base_data", "USAN_Stems.xlsx")
    usan_file_new = os.path.join(os.getcwd(), "base_data", "USAN_Stems.csv")

    df = pd.read_excel(usan_file)
    df = df[["Prefix (xxx-), Infix (-xxx-), or Suffix (-xxx)"]]
    df = df.dropna(subset=["Prefix (xxx-), Infix (-xxx-), or Suffix (-xxx)"])
    df["Stems"] = df["Prefix (xxx-), Infix (-xxx-), or Suffix (-xxx)"].apply(lambda x: x.split(","))
    df = df.explode(column=["Stems"])
    df["Stems"] = df["Stems"].apply(lambda x: x.split(" "))
    df = df.explode(column=["Stems"])
    df = df[(df["Stems"].str.startswith("-")) | (df["Stems"].str.endswith("-"))]
    df["Stems"] = df["Stems"].apply(lambda x: [x] if not ("(" in x or ")" in x) else remove_parentheses(x))
    df = df.explode(column=["Stems"])


    df = df[["Stems"]].drop_duplicates()
    df.to_csv(usan_file_new)

def remove_parentheses(x):
    s = x.find("(")
    e = x.find(")")
    return [f"{x[:s]}{x[e+1:]}", f"{x[:s]}{x[s+1:e]}{x[e+1:]}"]


if __name__ == "__main__":
    prep_usan()