import json

from sklearn.metrics import cohen_kappa_score


def main():
    import numpy as np
    annot_kappa = []
    skipped = 0
    i = 0
    dataset = {}
    annotator1 = json.load(open("./dataset/annotator1.json", "r"))
    annotator2 = json.load(open("./dataset/annotator2.json", "r"))
    students = json.load(open("./dataset/student.json", "r"))

    for annot1, annot2 in zip(annotator1, annotator2):
        annot1_label = annot1["labels"]
        annot2_label = annot2["labels"]
        if i == 22:
            breakpoint()
        i += 1
        try:
            kappa = cohen_kappa_score(annot1_label, annot2_label, labels=["I-PERIOD", "I-COMMA", "O"])
        except ValueError:
            skipped += 1
            print(annot1["text_id"], annot2["text_id"])
            print(len(annot1['labels']), len(annot2['labels']))
            continue
        if kappa is not np.nan:
            annot_kappa.append(kappa)
    breakpoint()


if __name__ == '__main__':
    main()
