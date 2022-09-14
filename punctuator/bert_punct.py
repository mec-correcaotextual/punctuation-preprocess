import json
import os
import re
import string
import traceback
from itertools import chain

import numpy as np
import spacy
import torch
from nltk.tokenize import wordpunct_tokenize
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


def truncate_sentences(text, max_seq_length=512, overlap=20):
    texts = []
    tokens = bert_tokenizer.tokenize(text)

    if len(tokens) > max_seq_length:
        if len(tokens) % max_seq_length != 0:
            max_seq_length //= 2

        for i in range(0, len(tokens), max_seq_length):
            slide = 0 if i == 0 else overlap
            truncated_tokens = tokens[i - slide:i + max_seq_length]
            new_text = bert_tokenizer.convert_tokens_to_string(truncated_tokens)

            texts.append(new_text)

        if len(bert_tokenizer.tokenize(texts[-1])) + len(bert_tokenizer.tokenize(texts[-2])) < max_seq_length:
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
    text = remove_punctuation(text)
    paragraphs = truncate_sentences(text, 256)

    return paragraphs


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
        # Tokenização do BERT tá diferente daque é feita aqui

        tokens = wordpunct_tokenize( text.lower())
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


if __name__ == '__main__':

    model = get_model(MODEL_PATH, model_type="bert", max_seq_length=512)

    DATA_PATH = "../dataset/"
    for filename in os.listdir(DATA_PATH):
        bert_labels = []
        true_labels = []
        kappa = []
        bert_annotator = []
        annotator_entities = json.load(open(os.path.join(DATA_PATH, filename), "r"))
        for item in annotator_entities:
            text_id = item["text_id"]
            print("Processing Text ID: ", text_id)
            ann_text = item["text"].lower()
            if text_id != 156:
                continue
            bert_label = predict(ann_text, model)
            true_label = text2labels(ann_text)
            true_labels.append(true_label)
            bert_labels.append(bert_label)
            kappa_score = cohen_kappa_score(true_label, bert_label, labels=["O", "I-COMMA", "I-PERIOD"])
            kappa.append(kappa_score)
            print("Kappa score: ", kappa_score)
            print("-" * 150)

            item.pop("ents")
            item.pop("labels")
            bert_annotation = item
            bert_annotation["labels"] = bert_label
            bert_annotation["cohen_kappa"] = kappa_score
            bert_annotator.append(bert_annotation)
        bert_annotator.append({
            "cohen_kappa": float(np.mean(kappa)),
            "cohen_kappa_std": float(np.std(kappa)),
            "report": classification_report(true_labels, bert_labels, output_dict=True)
        })
        print("Mean Kappa score: ", np.mean(kappa))
        os.makedirs("bert_annotations", exist_ok=True)
        with open(os.path.join("bert_annotations", "bert_" + filename), "w") as f:
            json.dump(bert_annotator, f, indent=4, cls=NpEncoder)

    breakpoint()
