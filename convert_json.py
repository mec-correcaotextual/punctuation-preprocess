import json
from typing import Literal
import srsly
from spacy.language import Language
from spacy import tokens
import spacy
import pathlib
import re

nlp = spacy.blank('pt')


def get_gold_token(text, start_char, tokens_delimiters=None):
    if tokens_delimiters is None:
        tokens_delimiters = [' ', '\n', '\t']

    limit_chars = text[:start_char]
    gold_token = []
    end_char = start_char
    for char in reversed(limit_chars):

        if char in tokens_delimiters and start_char != end_char:
            break
        else:
            start_char -= 1
            gold_token.append(char)

    gold_token = ''.join(reversed(gold_token))
    return start_char, end_char, gold_token


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

        for text_id, zipped_anot_data in enumerate(annotated_pairs, start=1):

            # verifica se os textos entre os anotadores são iguais
            assert all(zipped_anot_data[0]['text'] == d['text'] for d in zipped_anot_data), \
                'A sequencia de textos está diferente. por tanto não é possível comparar'

            # cria o documento pelo texto do primeiro anotador
            doc = nlp.make_doc(zipped_anot_data[0]['text'])
            # doc.user_data = {'student': '...'} # é possível adicionar informações dos alunos ao dataset
            text = zipped_anot_data[0]['text'].replace('\n', ' ').replace('\t', ' ')

            end_chars = []
            student_entity = {
                'text': text,
                'sentence_id': text_id,
                'ents': []
            }
            annotator_entity = {
                'text': text,
                'sentence_id': text_id,
                'ents': []
            }

            for annotator_id, annotation in enumerate(zipped_anot_data, start=1):
                # criando spans a partir dos indices
                errors_filter = ['Erro de Pontuação', 'Erro de Vírgula']

                for s in annotation['label']:

                    if len(annotation['label']) > 2 and s[2] in errors_filter:
                        new_span = doc.char_span(*s, alignment_mode=token_alignment)

                        if new_span is None or new_span.start_char == new_span.end_char:
                            start_char, end_char, gold_token = get_gold_token(text, s[1])
                            new_span = doc.char_span(*(start_char, end_char), alignment_mode=token_alignment)

                        if all(start < new_span.start_char for start in end_chars):
                            end_chars.append(new_span.end_char)

                            if text[s[0]:s[1]] == '.':
                                student_entity['ents'].append(
                                    {"start": new_span.start_char, "end": new_span.end_char, "label": "PERIOD"})
                            elif text[s[0]:s[1]] == ',':
                                student_entity['ents'].append(
                                    {"start": new_span.start_char, "end": new_span.end_char, "label": "COMMA"})

                            if s[2] == 'Erro de Pontuação':
                                if text[s[0]:s[1]] != '.':
                                    annotator_entity['ents'].append(
                                        {"start": new_span.start_char, "end": new_span.end_char, "label": "PERIOD"})
                            elif s[2] == 'Erro de Vírgula':
                                if text[s[0]:s[1]] != ',':
                                    student_entity['ents'].append(
                                        {"start": new_span.start_char, "end": new_span.end_char, "label": "COMMA"})
                                else:
                                    student_entity['ents'].append(
                                        {"start": new_span.start_char, "end": new_span.end_char, "label": "I-COMMA"})
                        else:

                            overlaps += 1
            student_entities.append(student_entity)
            annotator_entities.append(annotator_entity)
        print("overlaps: ", overlaps)
    return student_entities, annotator_entities


if __name__ == '__main__':
    sts_entities, annot_entities = convert_annotations('Anotation/')
    json.dump(obj=sts_entities, fp=open('student_entities.json', 'w'), indent=4)
    json.dump(obj=annot_entities, fp=open('annotator_entities.json', 'w'), indent=4)
