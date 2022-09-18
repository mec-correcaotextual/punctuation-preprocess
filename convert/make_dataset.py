import json
import pathlib
from collections import defaultdict
from typing import Literal

import srsly

from utils import text2labels, find_token_span
from utils.preprocess import preprocess_text, fix_break_lines


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


def remove_extra_punctuation(ann_text_list, start_char):
    """Remove pontuação extra
    :param ann_text_list: lista de caracteres do texto
    :param start_char: índice do caracter a partir do qual a busca será feita
    :return: lista de caracteres do texto
    """
    shift = 0
    i = start_char
    char = ann_text_list[start_char]

    while char in [' ', '\n', '\t', '.', ',', ';', ':', '!', '?']:
        ann_text_list.pop(i)
        shift -= 1

        char = ann_text_list[i]

    return ann_text_list, shift


def define_char_case(punct, text_list, i):
    """Define se o caracter é maiúsculo ou minúsculo"""

    if punct == '.':
        text_list[i + 2] = text_list[i + 2].upper()
        # Coloca aprimeira letra em maiúsculo
    elif punct == ',':
        text_list[i + 2] = text_list[i + 2].lower()
        # Coloca a primeira letra em minúsculo
    return text_list


def fix_punctuation(ann_text_list, start_char, end_char, punct):
    other_punctuations = ['.', ',', ';', ':', '!', '?']
    other_punctuations.remove(punct)
    shift = 0
    try:
        text_span = ann_text_list[start_char:end_char][0]
    except IndexError:
        # Não há matches com o caracter do texto e então significa que o aluno esqueceu ponto final.
        if punct == '.':
            ann_text_list.append('.')
            shift += 1
        return ann_text_list, shift

    if text_span != punct:

        try:
            for i in range(start_char - 3, end_char + 3):
                old_char = ann_text_list[i]
                if old_char in other_punctuations:
                    ann_text_list[i] = punct
                    ann_text_list = define_char_case(punct, ann_text_list, i)
                    break
                if old_char in [' ', '\n', '\t']:
                    ann_text_list[i] = punct

                    ann_text_list.insert(i + 1, old_char)
                    shift += 1
                    ann_text_list, shift_removed = remove_extra_punctuation(ann_text_list, i + 2)
                    shift += shift_removed
                    ann_text_list = define_char_case(punct, ann_text_list, i)
                    break
        except IndexError:
            ann_text_list.append(punct)
            shift += 1


    else:

        for i in range(start_char - 1, end_char + 1):
            if ann_text_list[i] == punct:
                ann_text_list.pop(i)  # Remove pontuação extra
                break
        shift -= 1

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

            student_entity["ents"] = find_token_span(new_sts_text, token_alignment=token_alignment)
            student_entity["labels"] = text2labels(student_entity["text"])

            for annotator_id, annotation in enumerate(zipped_anot_data, start=1):
                global_shift = 0
                text = annotation['text']
                len_b = len(list(text))
                text = fix_break_lines(text)

                ann_text_list = list(text)
                len_a = len(ann_text_list)
                if len_a != len_b:
                    global_shift = len_a - len_b

                for ann_span in annotation['label']:
                    local_shift = 0

                    start_char, end_char, label = ann_span[1] - 1 + global_shift, ann_span[1] + global_shift, ann_span[
                        2]
                    # Descobrir o porquê há multiplas pontuações no texto do aluno e corrigir isso 'esta podre.?,
                    if label not in ['Erro de Pontuação', 'Erro de vírgula']:
                        continue

                    symbol = '.' if label == 'Erro de Pontuação' else ','

                    ann_text_list, local_shift = fix_punctuation(ann_text_list, start_char, end_char, punct=symbol)

                    global_shift += local_shift

                atitle, new_ann_textp = preprocess_text(''.join(ann_text_list))
                new_ann_text = '\n'.join(new_ann_textp)
                after_labels = text2labels(new_ann_text)

                annotator_entity[annotator_id]["text"] = new_ann_text
                annotator_entity[annotator_id]["title"] = title
                annotator_entity[annotator_id]["ents"] = find_token_span(new_ann_text, token_alignment=token_alignment)
                annotator_entity[annotator_id]["labels"] = after_labels
                student_entities.append(student_entity)
            annotator_entities.append(annotator_entity)

    return student_entities, annotator_entities


if __name__ == '__main__':
    sts_entities, annot_entities = convert_annotations('../annotations/')
    annotator1 = list(map(lambda dict_annot: dict_annot[1], annot_entities))
    annotator2 = list(map(lambda dict_annot: dict_annot[2], annot_entities))

    json.dump(obj=sts_entities, fp=open('../dataset/student.json', 'w'), indent=4)
    json.dump(obj=annotator1, fp=open('../dataset/annotator1.json', 'w'), indent=4)
    json.dump(obj=annotator2, fp=open('../dataset/annotator2.json', 'w'), indent=4)
