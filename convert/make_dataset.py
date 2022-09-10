import json
import pathlib
import string
from collections import defaultdict
from typing import Literal

import srsly

from utils import text2labels, find_token_span
from utils.preprocess import preprocess_text


def remove_space_before_punct(text_list):
    """
    Remove espaços antes de pontuação
    :param text_list: Lista de caracteres do texto
    :return:
    """
    shift = 0
    for i in range(len(text_list)):
        if text_list[i] in string.punctuation:
            if text_list[i-1] in [' ', '\n', '\t']:
                text_list.pop(i-1)
                shift -= 1
    return text_list, shift


def remove_repeated_punctuation(text_list, start_char, ref_punct):
    """Remove pontuação repetida"""
    shift = 0
    for j in range(start_char, len(text_list)):
        if text_list[j] != ref_punct:
            text_list.pop(j)
            shift -= 1
        if text_list[j] in [' ', '\n', '\t']:
            break

    return text_list, shift


def fix_punctuation(sts_text_list, ann_text_list, start_char, end_char, punct):
    opposite = '.' if punct == ',' else ','
    shift = 0
    try:
        text_span = sts_text_list[start_char:end_char][0]

        if text_span != punct:

            if text_span == opposite:
                for i in range(start_char, len(sts_text_list)):
                    if ann_text_list[i] == opposite:
                        ann_text_list[i] = punct
                        break

                ann_text_list, shift = remove_space_before_punct(ann_text_list)

            else:
                try:
                    for i in range(start_char - 1, end_char + 1):
                        old_char = ann_text_list[i]
                        if old_char in [' ', '\n', '\t']:
                            ann_text_list[i] = punct

                            ann_text_list.insert(i + 1, old_char)

                            ann_text_list[i + 2] = ann_text_list[
                                i + 2].upper()  # Coloca aprimeira letra em maiúsculo
                            shift += 1
                            break
                except IndexError:
                    ann_text_list.append(punct)

        else:
            for i in range(start_char - 1, end_char + 1):
                if ann_text_list[i] in string.punctuation:
                    ann_text_list.pop(i)  # Remove pontuação extra
            shift -= 1
    except IndexError:
        # Não há matches com o caracter do texto e então significa que o aluno esqueceu ponto final.
        if punct == '.':
            ann_text_list.append('.')
            shift += 1
    return ann_text_list, shift


def convert_annotations(
        path: str = 'data',
        token_alignment: Literal['contract', 'expand'] = 'expand'
):
    """Converte jsonl do doccano para o estilo de anotação do SpaCy e retorna um docbin com todos dos docs"""
    result = [p for p in pathlib.Path(path).glob('./**/Anotações')]
    result.sort()

    annotator_entities = []
    student_entities = []

    for week_path in result:
        print(week_path)
        overlaps = 0
        # paths -> json generators -> list[json]
        jsonls = list(week_path.glob('anot*'))
        jsonls.sort()

        annotated_jsonl = [tuple(srsly.read_jsonl(path)) for path in jsonls]
        annotated_pairs = tuple(zip(*annotated_jsonl))

        for i, zipped_anot_data in enumerate(annotated_pairs, start=1):

            text = zipped_anot_data[0]['text']
            text_id = zipped_anot_data[0]['id']
            title, new_sts_text = preprocess_text(text)
            new_sts_text = '\n'.join(new_sts_text)

            student_entity = {'text': new_sts_text, "title": title, 'text_id': text_id, 'ents': []}

            annotator_entity = defaultdict(lambda: {'text': text, "title": title, 'text_id': text_id, 'ents': []})

            # Procura pela pontuação do aluno no texto

            student_entity["ents"] = find_token_span('\n'.join(new_sts_text), token_alignment=token_alignment)
            student_entities.append(student_entity)

            sts_text_list = list(text)  # Lista de caracteres do texto do aluno

            for annotator_id, annotation in enumerate(zipped_anot_data, start=1):
                shifts = 0

                ann_text_list = list(annotation['text'])

                for s in annotation['label']:
                    shift = 0

                    start_char, end_char, label = s[1] - 1 + shifts, s[1] + shifts, s[2]

                    if label == 'Erro de Pontuação':

                        ann_text_list, shift = fix_punctuation(sts_text_list, ann_text_list, start_char, end_char,
                                                               punct='.')

                    elif label == 'Erro de vírgula':

                        ann_text_list, shift = fix_punctuation(sts_text_list, ann_text_list, start_char, end_char,
                                                               punct=',')
                    shifts += shift

                atitle, new_ann_textp = preprocess_text(''.join(ann_text_list))
                new_ann_text = '\n'.join(new_ann_textp)
                annotator_entity[annotator_id]["text"] = new_ann_text
                annotator_entity[annotator_id]["title"] = title
                annotator_entity[annotator_id]["ents"] = find_token_span(new_ann_text, token_alignment=token_alignment)
                annotator_entity[annotator_id]["labels"] = text2labels(new_ann_text)
            annotator_entities.append(annotator_entity)

    return student_entities, annotator_entities


if __name__ == '__main__':
    sts_entities, annot_entities = convert_annotations('../annotations/')
    annotator1 = list(map(lambda dict_annot: dict_annot[1], annot_entities))
    annotator2 = list(map(lambda dict_annot: dict_annot[2], annot_entities))

    json.dump(obj=sts_entities, fp=open('../dataset/student_entities.json', 'w'), indent=4)
    json.dump(obj=annotator1, fp=open('../dataset/annotator1_entities.json', 'w'), indent=4)
    json.dump(obj=annotator2, fp=open('../dataset/annotator2_entities.json', 'w'), indent=4)
