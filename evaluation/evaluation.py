import json
import string
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import seqeval
from nltk import wordpunct_tokenize
from seqeval.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import spacy

nlp = spacy.load("pt_core_news_lg")


def get_word_label_dict(sentence, labels):
    tokens = [token.lower() for token in wordpunct_tokenize(sentence) if token not in string.punctuation]
    word_label_dict = defaultdict(list)
    for word, label in zip(tokens, labels):
        word_label_dict[label].append(word)
    return word_label_dict


def get_words_statistics(dataset):
    word_stats = {}
    for data_name in dataset:
        data = dataset[data_name]
        punkt_words = defaultdict(list)
        word_stats[data_name] = punkt_words
        for item in data:

            try:
                word_dict = get_word_label_dict(item["text"], item["labels"])

                for key in word_dict.keys():
                    word_stats[data_name][key].extend(word_dict[key])
            except KeyError:
                print(item)

    return word_stats


def get_cohen_statistics(annotator1, annotator2):
    annot_kappa = []

    dataset1_empty_labels = 0
    dataset2_empty_labels = 0
    skip = False
    value_erros = 0
    for ann1, ann2 in zip(annotator1, annotator2):
        annot1_label = ann1["labels"]
        annot2_label = ann2["labels"]

        if len(list(set(annot1_label))) == 1 and list(set(annot1_label))[0] == "O":
            dataset1_empty_labels += 1
            skip = True
        if len(list(set(annot2_label))) == 1 and list(set(annot2_label))[0] == "O":
            dataset2_empty_labels += 1
            skip = True
        if skip:
            skip = False
            continue
        try:
            kappa = cohen_kappa_score(annot1_label, annot2_label, labels=["I-PERIOD", "I-COMMA", "O"])
            annot_kappa.append(kappa)
        except ValueError:
            value_erros += 1

    print("skipped to missmatch labels", value_erros)
    skipped = dataset2_empty_labels + dataset1_empty_labels
    statistics = {
        "skipped": skipped,
        "dataset2_empty_labels": dataset2_empty_labels,
        "dataset1_empty_labels": dataset1_empty_labels,
        "kappa_mean": np.mean(annot_kappa),
        "kappa_std": np.std(annot_kappa),
        "kappa_min": np.min(annot_kappa),
        "kappa_max": np.max(annot_kappa),
        "kappa_median": np.median(annot_kappa),
        "kappa_25": np.percentile(annot_kappa, 25),
        "kappa_75": np.percentile(annot_kappa, 75),
        "kappa_90": np.percentile(annot_kappa, 90),
        "kappa_95": np.percentile(annot_kappa, 95),
        "kappa_99": np.percentile(annot_kappa, 99),
        "total_annotations": len(annotator1)
    }
    return statistics


def dataset_comparasion(dataset):
    statistics = {

    }

    data_names = list(dataset.keys())
    comb = combinations(data_names, 2)

    for i, (data_name1, data_name2) in enumerate(comb):
        data1 = dataset[data_name1]
        data2 = dataset[data_name2]

        statistics[f"{data_name1}_{data_name2}"] = get_cohen_statistics(data1, data2)

    return pd.DataFrame.from_dict(statistics, orient="index").T.round(3)


def main():
    annotator1 = json.load(open("../dataset/annotator1.json", "r"))
    annotator2 = json.load(open("../dataset/annotator2.json", "r"))
    bertannotation = json.load(open("../punctuator/bert_annotations/bert_annotator2.json", "r"))
    both_annotator = json.load(open("../dataset/both_anotators.json", "r"))
    dataset = {
        "annotator1": annotator1,
        "annotator2": annotator2,
        "bertannotation": bertannotation
    }
    statistics = dataset_comparasion(dataset)
    words_sts = get_words_statistics(dataset)
    print(words_sts["annotator1"]["I-PERIOD"][:10])
    print(words_sts["annotator2"]["I-PERIOD"][:10])
    print(words_sts["annotator1"]["I-COMMA"][:10])
    print(words_sts["annotator1"]["O"][:10])

    stats = {
        "annotator1": {
            "I-PERIOD": len(words_sts["annotator1"]["I-PERIOD"]),
            "I-COMMA": len(words_sts["annotator1"]["I-COMMA"]),

        },
        "annotator2": {
            "I-PERIOD": len(words_sts["annotator2"]["I-PERIOD"]),
            "I-COMMA": len(words_sts["annotator2"]["I-COMMA"]),
        },
    }

    pd.DataFrame.from_dict(stats, orient="index").T.round(3).to_csv("stats.csv")

    print(Counter(words_sts["annotator1"]["I-PERIOD"]))
    print(Counter(words_sts["annotator2"]["I-PERIOD"]))
    print(len(words_sts["annotator2"]["I-PERIOD"]))
    true_labels = []
    pred_labels = []
    for i in range(len(both_annotator)):

        if len(both_annotator[i]["labels"]) == len(bertannotation[i]["labels"]):
            true_labels.extend(both_annotator[i]["labels"])
            pred_labels.extend(bertannotation[i]["labels"])
        else:
            print("missmatch", i)

    print(true_labels[:1])
    print(pred_labels[:1])
    report = classification_report(true_labels, pred_labels)
    # report_df = pd.DataFrame(report).T.round(3)
    # report_df.to_csv("report.csv")
    print(report)
    statistics.to_csv("annotator2_bertannotation.csv", index_label="metrics")


if __name__ == '__main__':
    main()
