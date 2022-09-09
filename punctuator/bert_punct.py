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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
silence_tensorflow()
from simpletransformers.ner import NERModel, NERArgs
from transformers import BertTokenizer

nlp = spacy.blank('pt')
MODEL_PATH = "../models/bert-portuguese-tedtalk2012"
tokenizer = regexp.RegexpTokenizer(r'\w+|[.,?!]')
tokenizer_words = regexp.RegexpTokenizer(r'\w+')
bert_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


def text2labels(sentence):
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


def preprocess_text(text, max_seq_length):
    texts = []

    clean_text = ' '.join(tokenizer_words.tokenize(text))
    tokens = bert_tokenizer.tokenize(clean_text)

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

    return texts


def predict(text, model, max_seq_length):
    texts = preprocess_text(text, max_seq_length)
    prediction_list, raw_outputs = model.predict(texts)
    pred_dict = merge_dicts(list(chain(*prediction_list)))
    return pred_dict


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

    annotator_entities = json.load(open("../dataset/annotator1_entities.json", "r"))

    model = get_model(MODEL_PATH, model_type="bert", max_seq_length=512)
    bert_labels = []
    true_labels = []

    for item in annotator_entities:

        # TODO Separar o texto em parágrafos com \n
        text_id = item["text_id"]

        words = [word for word in word_tokenize(item["text"]) if word not in string.punctuation]
        test_text = ' '.join(words).lower()
        predictions = predict(test_text, model, 512)
        true_label = text2labels(item["text"])
        bert_label = get_labels(test_text, predictions)

        if len(bert_label) != len(true_label):
            print("Tamanho diferente")
            print(len(words))
            print(words)

            print(len(bert_label), len(true_label), len(item["labels"]))
            print(bert_label)
            print(true_label)
            print(test_text)
            breakpoint()
        true_labels.append(true_label)
        bert_labels.append(bert_label)

    report = metrics.classification_report(true_labels, bert_labels, output_dict=True)
    pd.DataFrame.from_dict(report, orient='index').to_csv(f"results.csv")
    print(report)
    breakpoint()