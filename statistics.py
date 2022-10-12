import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

ANNOTATOR_ID = 1


def main():
    reports = {
        1: defaultdict(float),
        2: defaultdict(float)
    }

    BERT_PATH = f"bert_annotations/annotator{ANNOTATOR_ID}/reports"
    for file in os.listdir(BERT_PATH):
        per = re.findall(r"\d\.\d", file)[0]
        annot = int(re.findall(r"\d", file)[0])

        report = pd.read_csv(os.path.join(BERT_PATH, file))
        report.set_index("metrics", inplace=True)
        report = report.T

        reports[annot][per] = report["f1-score"]['micro avg']

    df = pd.DataFrame.from_dict(dict(reports[ANNOTATOR_ID]), orient='index')
    print(df.mean(axis=1))
    df.plot(kind='line', title=f"Annotator {ANNOTATOR_ID}",
            xlabel="Cohen's kappa", ylabel="F1-score")

    plt.show()


if __name__ == "__main__":
    main()
