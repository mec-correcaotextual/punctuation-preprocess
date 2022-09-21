import json
import pathlib
from collections import defaultdict
from typing import Literal

import srsly

from convert.util import fix_punctuation
from utils import text2labels, find_token_span
from utils.preprocess import preprocess_text, fix_break_lines


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

            annotator_entity = {'text': text, "title": title, 'text_id': text_id, 'ents': []}

            # Procura pela pontuação do aluno no texto

            student_entity["ents"] = find_token_span(new_sts_text, token_alignment=token_alignment)
            student_entity["labels"] = text2labels(student_entity["text"])

            text = student_entity['text']
            len_b = len(list(text))
            text = fix_break_lines(text)
            global_shift = 0
            ann_text_list = list(text)
            len_a = len(ann_text_list)
            if len_a != len_b:
                global_shift = len_a - len_b
            for annotator_id, annotation in enumerate(zipped_anot_data, start=1):

                for ann_span in annotation['label']:

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

            annotator_entity["text"] = new_ann_text
            annotator_entity["title"] = title
            annotator_entity["ents"] = find_token_span(new_ann_text, token_alignment=token_alignment)
            annotator_entity["labels"] = after_labels
            student_entities.append(student_entity)
            annotator_entities.append(annotator_entity)

    return student_entities, annotator_entities


if __name__ == '__main__':
    sts_entities, annot_entities = convert_annotations('../annotations/')

    json.dump(obj=sts_entities, fp=open('../dataset/student.json', 'w'), indent=4)
    json.dump(obj=annot_entities, fp=open('../dataset/both_anotators.json', 'w'), indent=4)
