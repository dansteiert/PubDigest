import logging
import seaborn as sns
import matplotlib.pyplot as plt
from misc.save_files import read_pickle



df = read_pickle("D:\Charite\PubMedCrawler\data\chronic thromboembolic pulmonary hypertension\Data_for_Plotting\Embedding\_med\\retrospective_validation_skip-gram_2010_CTEPH_venous thrombosis.pkl", logger=logging.getLogger())

df_groupby = df.groupby(by=["query_term"])


df_temp = df_groupby.apply(lambda x: correct_predictions(x))
df_temp.sort_values(by="similarity", inplace=True)

sns.barplot(data=df_temp, x="similarity", y="query_term", color="blue")
plt.title("Retrospective Prediction - 2010\n"
          "Only True Predictions")
plt.tight_layout()
plt.show()



def correct_predictions(groupby_row):
    before = groupby_row[groupby_row["year"] == "before_prediction"]["co_occurrence"] == 0
    if before.values[0]:
        after = groupby_row[groupby_row["year"] == "after_prediction"]["co_occurrence"] != 0
        if after.values[0]:
            return groupby_row[groupby_row["year"] == 2010]