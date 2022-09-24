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


def select_subsamples(dataset, minimum_cohen_kappa=0.5):
    new_dataset = []
    data_names = list(dataset.keys())
    comb = combinations(data_names, 2)

    for data_name1, data_name2 in comb:
        subsamples = defaultdict(list)
        for item1, item2 in zip(dataset[data_name1], dataset[data_name2]):
            try:
                kappa_score = cohen_kappa_score(item1["labels"], item2["labels"], labels=["I-PERIOD", "I-COMMA", "O"])
            except ValueError:
                breakpoint()
            if kappa_score >= minimum_cohen_kappa:
                subsamples[data_name1].append(item1)
                subsamples[data_name2].append(item2)
        new_dataset.append(subsamples)
    return new_dataset


def check_if_empty(labels):
    if len(list(set(labels))) == 1 and list(set(labels))[0] == "O":
        return True
    return False


def get_valid_dataset(dataset):
    valid_labels = defaultdict(list)
    datanames = list(dataset.keys())

    for annot1, annot2, annot3 in zip(dataset[datanames[0]], dataset[datanames[1]], dataset[datanames[2]]):

        if any([check_if_empty(annot1["labels"]), check_if_empty(annot2["labels"]),
                check_if_empty(annot3["labels"])]):
            continue
        if len(annot1["labels"]) != len(annot2["labels"]) or len(annot1["labels"]) != len(annot3["labels"]):
            breakpoint()
        for data_label, item in [(datanames[0], annot1["labels"]), (datanames[1], annot2["labels"]),
                                 (datanames[2], annot3["labels"])]:
            valid_labels[data_label].append(item)

    return valid_labels


def get_cohen_statistics(data_dict):
    annot_kappa = []

    data_names = list(data_dict.keys())
    empty_labels = {
        data_names[0]: 0,
        data_names[1]: 0
    }

    skip = False
    value_erros = 0

    for ann1, ann2 in zip(data_dict[data_names[0]], data_dict[data_names[1]]):
        annot1_label = ann1["labels"]
        annot2_label = ann2["labels"]

        if len(list(set(annot1_label))) == 1 and list(set(annot1_label))[0] == "O":
            empty_labels[data_names[0]] += 1
            skip = True
        if len(list(set(annot2_label))) == 1 and list(set(annot2_label))[0] == "O":
            empty_labels[data_names[1]] += 1
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
    skipped = empty_labels[data_names[0]] + empty_labels[data_names[1]]
    statistics = {
        "skipped": skipped,
        f"{data_names[1]}_empty_labels": empty_labels[data_names[1]],
        f"{data_names[0]}_empty_labels": empty_labels[data_names[0]],
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
        "total_annotations": len(data_dict[data_names[0]])
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

        statistics[f"{data_name1}_{data_name2}"] = get_cohen_statistics({
            data_name1: data1,
            data_name2: data2
        })

    return pd.DataFrame.from_dict(statistics, orient="index").T.round(3)


def main():
    annotator1 = json.load(open("../dataset/annotator1.json", "r"))
    annotator2 = json.load(open("../dataset/annotator2.json", "r"))
    bertannotation = json.load(open("../dataset/bert_annotations.json", "r"))
    both_annotator = json.load(open("../dataset/both_anotators.json", "r"))
    dataset = {
        "annotator1": annotator1,
        "annotator2": annotator2,
        "bertannotation": bertannotation
    }
    statistics = dataset_comparasion(dataset)
    statistics.to_csv("statistics.csv", index_label="metrics")
    words_sts = get_words_statistics(dataset)

    valid_dataset = get_valid_dataset(dataset)

    report = classification_report(valid_dataset["annotator1"], valid_dataset["bertannotation"], output_dict=True)
    print(report)
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

    pd.DataFrame.from_dict(stats, orient="index").T.to_csv("stats.csv", index_label="metrics")
    dataset = {
        "annotator1": annotator1,
        "annotator2": annotator2, }
    subsamples = defaultdict(list)

    for per in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        new_dataset = select_subsamples(dataset, minimum_cohen_kappa=per)
        subsamples[per] = new_dataset
        print(len(new_dataset[0]["annotator1"]))
    json.dump(subsamples, open("subsamples.json", "w"), indent=4)
    # pd.DataFrame.from_dict(stats, orient="index").T.round(3).to_csv("punkt_stats.csv")
    #
    # print(Counter(words_sts["annotator1"]["I-PERIOD"]))
    # print(Counter(words_sts["annotator2"]["I-PERIOD"]))
    # print(len(words_sts["annotator2"]["I-PERIOD"]))
    # true_labels = []
    # pred_labels = []
    # for i in range(len(both_annotator)):
    #
    #     if len(both_annotator[i]["labels"]) == len(bertannotation[i]["labels"]):
    #         true_labels.extend(both_annotator[i]["labels"])
    #         pred_labels.extend(bertannotation[i]["labels"])
    #     else:
    #         print("missmatch", i)
    #
    # print(true_labels[:1])
    # print(pred_labels[:1])
    # report = classification_report(true_labels, pred_labels)
    # report_df = pd.DataFrame(report).T.round(3)
    # report_df.to_csv("report.csv")
    # print(report)


if __name__ == '__main__':
    main()
