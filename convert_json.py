import json
from collections import defaultdict
from typing import Literal
import srsly
from spacy.language import Language
from spacy import tokens
import spacy
import pathlib
import re

nlp = spacy.blank('pt')


def get_gold_token(text, start_char, end_char, tokens_delimiters=None, token_alignment='expand'):
    if tokens_delimiters is None:
        tokens_delimiters = [' ', '\n', '\t']
    doc = nlp.make_doc(text)
    new_span = doc.char_span(*(start_char, end_char), alignment_mode=token_alignment)

    if new_span is None or new_span.start_char == new_span.end_char:

        limit_chars = text[:start_char]
        gold_token = []
        end_char = start_char

        for char in reversed(limit_chars):

            if char in tokens_delimiters and start_char != end_char:
                break
            else:
                start_char -= 1
                gold_token.append(char)

        new_span = doc.char_span(*(start_char, end_char), alignment_mode=token_alignment)
        start_char, end_char = new_span.start_char, new_span.end_char
    else:
        start_char, end_char = new_span.start_char, new_span.end_char

    return start_char, end_char


pattern = re.compile(r'(?<=[a-z|A-z])[.,!?]')


def find_token_span(text):
    ents = []
    matches = re.finditer(pattern, text)
    for match in matches:
        span_start, span_end = match.span()

        non_puncted_text = re.sub('[.,!?]', ' ', text)
        start_char, end_char = get_gold_token(non_puncted_text, span_start, span_end, tokens_delimiters=None)
        if text[span_start:span_end] in ['.', '?', '!']:
            ents.append((start_char, end_char, "PERIOD"))
        elif text[span_start:span_end] in [',']:
            ents.append((start_char, end_char, "COMMA"))

    return ents


def drop_duplicates(annotation):
    new_annotation = []
    texts_ids = []
    for annot in annotation:
        if annot["text_id"] not in texts_ids:
            new_annotation.append(annot)
            texts_ids.append(annot["text_id"])
    return new_annotation


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

            student_entity["ents"] = find_token_span(text)
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
    sts_entities, annot_entities = convert_annotations('Anotation/')
    annotator1 = list(map(lambda dict_annot: dict_annot[1], annot_entities))
    annotator2 = list(map(lambda dict_annot: dict_annot[2], annot_entities))
    print(len(annotator1), len(annotator2), len(sts_entities))
    # annotator1 = drop_duplicates(annotator1)
    # annotator2 = drop_duplicates(annotator2)
    print(len(annotator1), len(annotator2), len(sts_entities))
    json.dump(obj=sts_entities, fp=open('student_entities.json', 'w'), indent=4)
    json.dump(obj=annotator1, fp=open('annotator1_entities.json', 'w'), indent=4)
    json.dump(obj=annotator2, fp=open('annotator2_entities.json', 'w'), indent=4)
