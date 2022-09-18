import json
import warnings
from sklearn.metrics import cohen_kappa_score

from utils import text2labels


def main():
    import numpy as np
    annot_kappa = []
    skipped = 0
    i = 0
    dataset = {}
    annotator1 = json.load(open("../dataset/annotator1.json", "r"))
    annotator2 = json.load(open("../dataset/annotator2.json", "r"))
    students = json.load(open("../dataset/student.json", "r"))

    for annot1, annot2 in zip(annotator1, annotator2):
        annot1_label = annot1["labels"]
        annot2_label = annot2["labels"]

        if len(annot1_label) != len(annot2_label):
            print("Skipping", annot1["text_id"])
            print("Annotator 1:", annot1["text"])
            print("Annotator 2:", annot2["text"])
            breakpoint()
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                kappa = cohen_kappa_score(text2labels(annot1["text"]), text2labels(annot2["text"]), labels=["I-PERIOD", "I-COMMA", "O"])
            except Warning:
                print(set(annot1_label), set(annot2_label))
                print(len(annot1_label), len(annot2_label))
                skipped += 1
                continue
        annot_kappa.append(kappa)

    print("Skipped: ", skipped)
    print("Total anntoation: ", len(annotator1))
    print("Mean kappa: ", np.mean(annot_kappa))
    print("Std kappa: ", np.std(annot_kappa))
    print("Max kappa: ", np.max(annot_kappa))
    print("Min kappa: ", np.min(annot_kappa))
    print("Median kappa: ", np.median(annot_kappa))




if __name__ == '__main__':
    main()
