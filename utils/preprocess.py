import json
import re

import spacy

special_pattern = r'\s+|\n+|/n|\t+'
marks = r'\[\w{0,3}|\W{0,3}\]|\}'


def join_split_words(text):
    """
    Junta palavras que foram separadas por um \n
    """

    splited = re.search(r'(\w+)-\n(\w+)', text)
    if splited:
        text = text.replace(splited.group(0), splited.group(1) + splited.group(2))
        return join_split_words(text)
    return text


def clean_text(text):
    """
    Remove caracteres especiais e espaços em branco e as
    marcações de início e fim de parágrafo e afins.
    :param text:
    :return:
    """
    text = re.sub(special_pattern, ' ', text)
    text = re.sub(marks, '', text)
    text = ' '.join(text.split()).replace('<i>', '').replace('</i>', '').replace('<i/>', '')
    text = re.sub(r'\.+', '.', text)
    return text


def split_paragraphs(text):
    """
    Separa o texto em parágrafos
    :param text:
    :return:
    """

    title = re.search(r'\[T\].*\n+', text)
    if title:
        title = title.group(0)
        text = text.replace(title, '')
        title = title.replace('[T]', '').replace('\n', '')
        title = ' '.join(title.split())
    else:
        title = ''

    return title, text.split('[P]')


def preprocess_text(text):
    text = join_split_words(text)
    title, paragraphs = split_paragraphs(text)
    paragraphs = [clean_text(p) for p in paragraphs]
    paragraphs = list(filter(lambda x: x != '', paragraphs))
    return title, paragraphs


def main():
    json_list = open("../annotations/Semana2/Anotações/anotador1.jsonl", "r", encoding="utf-8").readlines()
    nlp = spacy.blank("pt")
    for json_str in json_list:

        result = json.loads(json_str)
        if result['id'] == 161:
            text = result["text"]
            title, paragraphs = split_paragraphs(text)
            print(paragraphs[:1])
            paragraphs = [clean_text(p) for p in paragraphs]
            paragraphs = list(filter(lambda x: x != '', paragraphs))
            print(paragraphs[:1])
            shifts = 0
            for s in result['label']:
                start_char, end_char, label = s[0], s[1], s[2]
                if label == 'Erro de Pontuação' or label == 'Erro de vírgula':
                    doc = nlp.make_doc(text)
                    new_span = doc.char_span(*(start_char, end_char), alignment_mode='expand')
                    if new_span:
                        print('Texto: ', new_span.text)
                    print(label, new_span)
                    print(text[end_char-1:end_char])
                    if text[end_char-1:end_char] != ',':
                        for i in range((end_char-1) + shifts - 1, end_char + shifts + 1):
                            print(text[i], end='\t')

                    print('-------------------')


            break





if __name__ == '__main__':
    main()
