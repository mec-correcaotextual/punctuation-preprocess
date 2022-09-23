import string

from nltk.tokenize import regexp
from simpletransformers.ner import NERModel, NERArgs
import torch


def replace(sentence):
    tokenizer = regexp.RegexpTokenizer(r'\w+|[.,?]')

    tokens = tokenizer.tokenize(sentence.lower())

    labels = []
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                # sent_data.append([sent_id,'O',token])
                labels.append('O')
            elif token in ['.', '?']:
                # sent_data[-1][1] = 'PERIOD'
                labels[-1] = 'PERIOD'
            elif token == ',':
                # sent_data[-1][1] = 'COMMA'
                labels[-1] = 'COMMA'

        except IndexError:
            continue

    return labels


def load_model(path: str, labels: list, max_length: int = 512, model_type: str = 'bert'):
    model_args = NERArgs()
    model_args.labels_list = labels

    model_args.max_seq_length = max_length
    return NERModel(
        model_type,
        path,
        args=model_args,
        use_cuda=torch.cuda.is_available()
    )


def get_predicted_labels(model, sentence: str):
    predicted_labels = model.predict([sentence], )[0]

    y_pred = []
    for i, pred in predicted_labels:
        y_pred.append(list(map(lambda item: list(item.values())[0], pred)))
    return y_pred
