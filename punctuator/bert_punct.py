import json
import re
import string
import traceback
from itertools import chain

import pandas as pd
import spacy
import torch
from nltk.tokenize import regexp, word_tokenize
from seqeval import metrics
from silence_tensorflow import silence_tensorflow
import os
from nltk.tokenize import wordpunct_tokenize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
silence_tensorflow()
from simpletransformers.ner import NERModel, NERArgs
from transformers import BertTokenizer

nlp = spacy.blank('pt')
MODEL_PATH = "../models/bert-portuguese-tedtalk2012"

bert_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


def text2labels(sentence):
    """
    Convert text to labels
    :param sentence: text to convert
    :return:  list of labels
    """
    tokens = word_tokenize(sentence.lower())

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


def truncate_sentences(text, max_seq_length):
    texts = []
    tokens = bert_tokenizer.tokenize(text)

    if len(tokens) > max_seq_length:
        if len(tokens) % max_seq_length == 0:
            for i in range(0, len(tokens), max_seq_length):
                truncated_tokens = tokens[i:i + max_seq_length]
                new_text = bert_tokenizer.convert_tokens_to_string(truncated_tokens)

                texts.append(new_text)
        else:
            max_seq_length //= 2

            for i in range(0, len(tokens), max_seq_length):
                truncated_tokens = tokens[i:i + max_seq_length]
                new_text = bert_tokenizer.convert_tokens_to_string(truncated_tokens)

                texts.append(new_text)
            texts[-1] = texts[-1] + ' ' + texts.pop(-2)

    else:
        texts.append(text)


def split_paragraphs(text):
    paragraphs = text.split('\n')
    return paragraphs


def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = ' '.join(word for word in word_tokenize(text)
                    if word not in string.punctuation)
    return text


def preprocess_text(text):
    """
    Preprocess text for prediction
    :param text: text to preprocess
    :return:  list of preprocessed text
    """
    paragraphs = split_paragraphs(text)

    return list(map(lambda x: remove_punctuation(x), paragraphs))


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

    return get_labels(test_text, pred_dict)


def get_labels(text, pred_dict):
    labels = []
    try:
        ## Tokenização do BERT tá diferente daque é feita aqui
        for word in word_tokenize(text):
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
        print(text)
        breakpoint()
    return labels


if __name__ == '__main__':

    annotator_entities = json.load(open("../dataset/annotator2_entities.json", "r"))

    model = get_model(MODEL_PATH, model_type="bert", max_seq_length=512)
    bert_labels = []
    true_labels = []

    for item in annotator_entities:

        text_id = item["text_id"]
        ann_text = item["text"]
        bert_label = predict(ann_text, model)
        true_label = text2labels(item["text"])
        true_labels.append(true_label)
        bert_labels.append(bert_label)

    report = metrics.classification_report(true_labels, bert_labels, output_dict=True)
    pd.DataFrame.from_dict(report, orient='index').to_csv(f"results.csv")
    print(report)
    breakpoint()
