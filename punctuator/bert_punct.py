import json
import os
import re
import string
import traceback
from itertools import chain
import click
import numpy as np
import pandas as pd
import spacy
import torch
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from seqeval.metrics import classification_report
from silence_tensorflow import silence_tensorflow
from sklearn.metrics import cohen_kappa_score
from simpletransformers.ner import NERModel, NERArgs
from transformers import BertTokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
silence_tensorflow()

nlp = spacy.blank('pt')
MODEL_PATH = "../models/bert-portuguese-tedtalk2012"

bert_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


def text2labels(sentence):
    """
    Convert text to labels
    :param sentence: text to convert
    :return:  list of labels
    """
    tokens = wordpunct_tokenize(sentence.lower())

    labels = []
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                labels.append('O')
            elif token in ['.', '?', '!', ';']:
                labels[-1] = 'I-PERIOD'
            elif token == ',':
                labels[-1] = 'I-COMMA'

        except IndexError:
            raise ValueError(f"Sentence can't start with punctuation {token}")
    return labels


def merge_dicts(dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = dict_args[0]
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def get_model(model_path,
              model_type="bert",
              labels=None,
              max_seq_length=512):
    model_args = NERArgs()

    if labels is not None:
        model_args.labels_list = labels
    else:
        model_args.labels_list = ["O", "COMMA", "PERIOD", "QUESTION"]
    model_args.silent = True
    model_args.max_seq_length = max_seq_length
    return NERModel(
        model_type,
        model_path,
        args=model_args,
        use_cuda=torch.cuda.is_available()
    )


def tokenize_words(text, remove_punctuation=True):
    """
    Tokenize words in text
    :param remove_punctuation:
    :param text: text to tokenize
    :param remove_punctuation:  remove punctuation from text
    :return:  list of tokens
    """
    if remove_punctuation:
        words = [word for word in wordpunct_tokenize(text) if word not in string.punctuation]
    else:
        words = wordpunct_tokenize(text)
    return words


def truncate_sentences(text, max_seq_length=512, overlap=20):
    """
    Truncate sentences to fit into BERT's max_seq_length
    :param text:  text to truncate
    :param max_seq_length:  max sequence length
    :param overlap:  overlap between sentences
    :return:    list of truncated sentences
    """
    texts = []

    tokens = tokenize_words(text)

    bert_tokens = bert_tokenizer.tokenize(text)

    len_text = ((max_seq_length * len(tokens)) // len(bert_tokens)) + 1

    if len(bert_tokens) > max_seq_length:
        if len(tokens) % max_seq_length != 0:
            max_seq_length //= 2

        for i in range(0, len(tokens), len_text):
            slide = 0 if i == 0 else overlap
            truncated_tokens = tokens[i - slide:i + len_text]
            texts.append(' '.join(truncated_tokens))

        if len(tokenize_words(texts[-1])) + len(tokenize_words(texts[-2])) < len_text:
            texts[-2] = texts[-2] + texts[-1]
            texts.pop()

    else:
        texts.append(text)

    return texts


def split_lines(text):
    paragraphs = text.split('\n')
    return paragraphs


def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = ' '.join(word for word in wordpunct_tokenize(text)
                    if word not in string.punctuation)
    return text


def preprocess_text(text):
    """
    Preprocess text for prediction
    :param text: text to preprocess
    :return:  list of preprocessed text
    """
    paragraphs = truncate_sentences(text)

    return list(map(lambda x: remove_punctuation(x).lower(), paragraphs))


def predict(test_text: str, model):
    """
    Predict punctuation for text
    :param test_text:   text to predict punctuation for
    :param model:  model to use for prediction
    :return:  list of predicted labels
    """
    texts = preprocess_text(test_text)

    prediction_list, raw_outputs = model.predict(texts)
    pred_dict = merge_dicts(list(chain(*prediction_list)))
    words = tokenize_words(test_text)
    words = sorted(set(words))
    pred_words = sorted(pred_dict.keys())

    if len(pred_words) != len(words):
        print("Number of tokens doesn't match")
        print("Number of tokens in text: ", len(words))
        print("Number of tokens in prediction: ", len(pred_dict))
        breakpoint()
    return get_labels(test_text, pred_dict)


def get_labels(text, pred_dict):
    labels = []
    try:
        # Tokenização do BERT tá diferente daque é feita aqui

        tokens = wordpunct_tokenize(text.lower())

        for word in tokens:
            if word not in string.punctuation:
                if pred_dict[word] == "QUESTION":
                    label = "I-PERIOD"
                elif pred_dict[word] == "COMMA":
                    label = "I-COMMA"
                elif pred_dict[word] == "PERIOD":
                    label = "I-PERIOD"
                else:
                    label = "O"
                labels.append(label)
    except KeyError:
        print("KeyError", pred_dict)
        print(traceback.format_exc())
        print(len(bert_tokenizer.tokenize(text)))
        print(text)
        breakpoint()
    return labels


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@click.command()
@click.option('--text', '-t', help='Text to predict punctuation for')
def main(text=None):
    model = get_model(MODEL_PATH)
    labels = predict(text, model)
    print(labels)


if __name__ == '__main__':

    model = get_model(MODEL_PATH, model_type="bert", max_seq_length=512)

    DATA_PATH = "../dataset/"

    annotator1 = json.load(open("../dataset/annotator1.json", "r"))
    annotator2 = json.load(open("../dataset/annotator2.json", "r"))
    bert_annots = []
    both_annotator = json.load(open("../dataset/both_anotators.json", "r"))
    dataset = {
        "annotator1": annotator1,
        "annotator2": annotator2,
        "both_annotator": both_annotator
    }
    bert_labels = []

    for item in both_annotator:
        text_id = item["text_id"]
        print("Processing Text ID: ", text_id)
        if text_id != 583:
            continue
        ann_text = item["text"].lower()
        bert_label = predict(ann_text, model)

        bert_labels.append(bert_label)

        item.pop("ents")
        item.pop("labels")
        bert_annotation = item
        bert_annotation["labels"] = bert_label
        bert_annots.append(bert_annotation)
    with open("../dataset/bert_annotations.json", "w") as f:
        json.dump(bert_annots, f, cls=NpEncoder)

    for data_label in dataset:
        true_labels = []
        items = dataset[data_label]

        for item in items:
            true_labels.append(item["labels"])

        report = classification_report(true_labels, bert_labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(f"../dataset/{data_label}_bert_report.csv")
