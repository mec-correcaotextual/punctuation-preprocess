import re
import string
import json
import numpy as np
import spacy
from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize

nlp = spacy.blank('pt')
pattern = re.compile(r'(?<=[a-z|A-z])[.,!?]')


def check_mergebility(annot, ents):
    """
    Check if the annotation is mergeable with the previous annotation.
    :param ents: list of entities
    :param annot: annotation to be checked
    :return:  True if the annotation is mergeable with the previous annotation, False otherwise
    """
    merge = False
    if len(ents) == 0:
        return True
    else:
        for ent in ents:
            if ent[1] > ent[0] > annot[1]:
                merge = True
            elif ent[0] < ent[1] < annot[0]:
                merge = True
            else:
                merge = False
                break
    return merge


def drop_duplicates(annotation):
    new_annotation = []
    texts_ids = []
    for annot in annotation:
        if annot["text_id"] not in texts_ids:
            new_annotation.append(annot)
            texts_ids.append(annot["text_id"])
    return new_annotation


def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = [word.lower() for word in wordpunct_tokenize(text)
            if word not in string.punctuation]
    return text


def find_token_span(text, token_alignment='expand'):
    ents = []
    matches = re.finditer(pattern, text)
    for match in matches:
        span_start, span_end = match.span()

        non_puncted_text = re.sub('[.,!?]', ' ', text)
        start_char, end_char = get_gold_token(non_puncted_text, span_start, span_end,
                                              tokens_delimiters=None, token_alignment=token_alignment)
        if text[span_start:span_end] in ['.', '?', '!']:
            ents.append((start_char, end_char, "I-PERIOD"))
        elif text[span_start:span_end] in [',']:
            ents.append((start_char, end_char, "I-COMMA"))

    return ents


def get_gold_token(text, start_char, end_char, tokens_delimiters=None, token_alignment='expand'):
    """
    Get the token that corresponds to the gold annotation
    :param text:  text to get the token from
    :param start_char:  start character of the gold annotation
    :param end_char:    end character of the gold annotation
    :param tokens_delimiters:  delimiters of the tokens
    :param token_alignment:  alignment of the token
    :return:  start and end character of the token
    """
    print(start_char, end_char)
    if tokens_delimiters is None:
        tokens_delimiters = [' ', '\n', '\t']
    doc = nlp.make_doc(text)
    new_span = doc.char_span(*(start_char, end_char), alignment_mode=token_alignment)
    assert start_char <= len(text), f"End char {start_char} is bigger than text length {len(text)}"

    if new_span is None or new_span.start_char == new_span.end_char:

        limit_chars = text[:start_char]
        gold_token = []
        end_char = start_char

        for char in reversed(limit_chars):

            if char in tokens_delimiters and start_char != end_char:
                break
            else:
                if start_char > 0:
                    start_char -= 1
                else:
                    break
                gold_token.append(char)

        try:
            new_span = doc.char_span(*(start_char, end_char), alignment_mode=token_alignment)
            start_char, end_char = new_span.start_char, new_span.end_char
        except AttributeError:
            raise ValueError(f"Can't find token for {start_char}:{end_char}, the start end char exceded the text")

    else:
        start_char, end_char = new_span.start_char, new_span.end_char

    return start_char, end_char


def text2labels(sentence):
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
            print(sentence)
            print(tokens)
            raise ValueError(f"Sentence can't start with punctuation {token}")
    return labels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def main():
    text = "Olá, Mundo! Irei compra-los a dinheiro!"
    print(word_tokenize(text))

    print(text2labels(text))


if __name__ == '__main__':
    main()
