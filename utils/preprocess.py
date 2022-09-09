import json
import re

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
    json_list = open("../Anotation/Semana2/Anotações/anotador1.jsonl", "r", encoding="utf-8").readlines()
    for json_str in json_list:

        result = json.loads(json_str)
        if result['id'] == 161:
            text = result["text"]
            title, paragraphs = split_paragraphs(text)
            print(paragraphs[:1])
            paragraphs = [clean_text(p) for p in paragraphs]
            paragraphs = list(filter(lambda x: x != '', paragraphs))
            print(paragraphs[:1])
            print()




if __name__ == '__main__':
    main()
