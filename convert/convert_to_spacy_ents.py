import json
import pathlib
from collections import defaultdict
from typing import Literal

import srsly

from legacy_code.convert_text import nlp, get_gold_token
from utils import find_token_span, check_mergebility


def convert_annotations(
        path: str = 'data',
        token_alignment: Literal['contract', 'expand'] = 'expand'
):
    """Converte jsonl do doccano para o estilo de anotação do SpaCy e retorna um docbin com todos dos docs"""
    result = [p for p in pathlib.Path(path).glob('./**/Anotações')]
    result.sort()
    error_map = {
        'Esqueceu pontuação final [?!.]': 0,
        'Trocou pontuação final por vírgula': 0,
        'Esqueceu a vírgula': 0,
        'Trocou a vírgula por ponto final.': 0
    }

    annotator_entities = []
    student_entities = []
    print(result)
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
            student_entity = {'text': text, 'text_id': text_id, 'ents': []}

            annotator_entity = defaultdict(lambda: {'text': text, 'text_id': text_id, 'ents': []})
            doc = nlp.make_doc(zipped_anot_data[0]['text'])
            # Procura pela pontuação do aluno no texto

            student_entity["ents"] = find_token_span(text, token_alignment=token_alignment)
            student_entities.append(student_entity)

            for annotator_id, annotation in enumerate(zipped_anot_data, start=1):

                for s in annotation['label']:

                    if s[2] == 'Erro de Pontuação':
                        start_char, end_char = get_gold_token(text, s[0], s[1])
                        if text[s[0]:s[1]] != '.':
                            ent_span = (start_char, end_char, "PERIOD")
                            ann_ents = annotator_entity[annotator_id]["ents"]

                            if check_mergebility(ent_span, ann_ents):
                                annotator_entity[annotator_id]["ents"].append(ent_span)

                            if text[s[0]:s[1]] == ',':
                                error_map['Trocou pontuação final por vírgula'] += 1
                            else:
                                error_map['Esqueceu pontuação final [?!.]'] += 1

                    elif s[2] == 'Erro de vírgula':
                        start_char, end_char = get_gold_token(text, s[0], s[1])
                        ent_span = (start_char, end_char, "COMMA")
                        ann_ents = annotator_entity[annotator_id]["ents"]

                        if check_mergebility(ent_span, ann_ents):
                            annotator_entity[annotator_id]["ents"].append(ent_span)
                            if text[s[0]:s[1]] == '.':
                                error_map['Trocou a vírgula por ponto final.'] += 1
                            else:
                                error_map['Esqueceu a vírgula'] += 1

                ann_ents = annotator_entity[annotator_id]["ents"]

                for sts_ents in student_entity["ents"]:

                    if check_mergebility(sts_ents, ann_ents):
                        annotator_entity[annotator_id]["ents"].append(sts_ents)

            annotator_entities.append(annotator_entity)
    return student_entities, annotator_entities


if __name__ == '__main__':
    sts_entities, annot_entities = convert_annotations('../Anotation/')
    annotator1 = list(map(lambda dict_annot: dict_annot[1], annot_entities))
    annotator2 = list(map(lambda dict_annot: dict_annot[2], annot_entities))

    json.dump(obj=sts_entities, fp=open('../annotations/student_entities.json', 'w'), indent=4)
    json.dump(obj=annotator1, fp=open('../annotations/annotator1_entities.json', 'w'), indent=4)
    json.dump(obj=annotator2, fp=open('../annotations/annotator2_entities.json', 'w'), indent=4)
