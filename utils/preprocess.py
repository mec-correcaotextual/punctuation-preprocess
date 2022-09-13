import json
import re

import spacy

special_pattern = r'\s+|\n+|/n|\t+|-'
marks = r'\[\w{0,3}|\W{0,3}\]|\}'


def join_split_words(text):
    """
    Junta palavras separadas por um \n
    """

    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)_\n(\w+)', r'\1\2', text)
    return text


def fix_break_lines(text):
    text = re.sub(r'/n', '\n', text)
    return text


def separate_punctuation(text):
    text = re.sub(r'([.,?!;:])\w', r'\1 ', text)
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
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\.+', '.', text)
    return fix_break_lines(text)


def split_lines(text):
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

    return title, ' '.join(text.split()).split('\n')


def remove_space_before_punctuation(text):
    text = re.sub(r'\s+([.,?!;:])', r'\1', text)
    return text


def remove_extra_punctuation(text):
    text = re.sub(r'([.,?!;:])+', r'\1', text)
    return text


def preprocess_text(text):
    text = join_split_words(text)
    text = remove_space_before_punctuation(text)
    text = remove_extra_punctuation(text)
    text = separate_punctuation(text)
    title, lines = split_lines(text)
    lines = [clean_text(p) for p in lines]
    lines = list(filter(lambda x: x != '', lines))

    return title, lines


def main():
    json_list = open("../annotations/Semana2/Anotações/anotador1.jsonl", "r", encoding="utf-8").readlines()
    nlp = spacy.blank("pt")
    print(preprocess_text('esta podre.?, ela respondeu com um sorriso.'))
    breakpoint()
    for json_str in json_list:

        result = json.loads(json_str)
        if result['id'] == 162:
            text = result["text"]

            shifts = 0
            for s in result['label']:
                start_char, end_char, label = s[0], s[1], s[2]
                if label == 'Erro de Pontuação' or label == 'Erro de vírgula':
                    doc = nlp.make_doc(text)
                    new_span = doc.char_span(*(start_char, end_char), alignment_mode='expand')
                    if new_span:
                        print('Texto: ', new_span.text)
                    print(label, new_span)
                    print(text[end_char - 1:end_char])
                    if text[end_char - 1:end_char] != ',':
                        for i in range((end_char - 1) + shifts - 1, end_char + shifts + 1):
                            print(text[i], end='\t')

                    print('-------------------')

            break


if __name__ == '__main__':
    main()
