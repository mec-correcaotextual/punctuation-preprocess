import json
from typing import Literal
import numpy as np
from convert.util import fix_punctuation, read_data
from utils import text2labels, find_token_span, get_gold_token, remove_punctuation
from utils.preprocess import preprocess_text, fix_break_lines
import spacy

nlp = spacy.blank("pt")


def get_error_labels(text, labels, token_alignment='expand'):
    """Retorna as labels de erro de pontuação e vírgula"""

    new_span = get_gold_token(text, labels[0], labels[1])
    doc = nlp.make_doc(text)
    new_span = doc.char_span(*new_span, alignment_mode=token_alignment)

    return new_span





def convert_annotations(
        path: str = 'data',
        token_alignment: Literal['contract', 'expand'] = 'expand'
):
    """Converte jsonl do doccano para o estilo de anotação do SpaCy e retorna um docbin com todos dos docs"""

    annotator_entities = []
    student_entities = []

    annotated_pairs = read_data(path)

    for i, group_annottion in annotated_pairs.groupby('text_id'):
        text = group_annottion['text'].values[0]
        text_id = group_annottion['text_id'].values[0]

        title, new_sts_text = preprocess_text(text)
        new_sts_text = '\n'.join(new_sts_text)

        student_entity = {'text': new_sts_text, "title": title, 'text_id': text_id, 'ents': []}

        annotator_entity = {'text': text, "title": title, 'text_id': text_id, 'ents': []}

        # Procura pela pontuação do aluno no texto

        student_entity["ents"] = find_token_span(new_sts_text, token_alignment=token_alignment)
        student_entity["labels"] = text2labels(student_entity["text"])

        text = fix_break_lines(text)

        len_b = len(list(text))
        global_shift = 0
        ann_text_list = list(text)
        len_a = len(ann_text_list)
        if len_a != len_b:
            global_shift = len_a - len_b

        erros_labels = []

        for annotation in group_annottion['label'].values:

            for ann_span in annotation:

                start_char, end_char, label = ann_span[1] - 1 + global_shift, ann_span[1] + global_shift, ann_span[
                    2]
                # Descobrir o porquê há multiplas pontuações no texto do aluno e corrigir isso 'esta podre.?,
                if label not in ['Erro de Pontuação', 'Erro de vírgula']:
                    continue

                symbol = '.' if label == 'Erro de Pontuação' else ','
                ann_text_list, local_shift = fix_punctuation(ann_text_list, start_char, end_char, punct=symbol)
                global_shift += local_shift

                erros_labels.append((ann_span[0] - 1 + global_shift, ann_span[1] + global_shift, label))

        atitle, new_ann_textp = preprocess_text(''.join(ann_text_list))
        new_ann_text = '\n'.join(new_ann_textp)
        after_labels = text2labels(new_ann_text)

        e_labels = []
        for start_char, end_char, label in erros_labels:
            start_char, end_char = get_gold_token(''.join(ann_text_list), start_char, end_char)
            e_labels.append((start_char, end_char, label))

        annotator_entity["raw_text"] = ''.join(ann_text_list)
        annotator_entity["text"] = new_ann_text
        annotator_entity["title"] = title
        annotator_entity["labels"] = after_labels
        annotator_entity["error_labels"] = e_labels

        student_entities.append(student_entity)
        annotator_entities.append(annotator_entity)

    return student_entities, annotator_entities


if __name__ == '__main__':
    sts_entities, annot_entities = convert_annotations('../annotations/')

    json.dump(obj=sts_entities, fp=open('../datasets/full/student.json', 'w'), indent=4, cls=NpEncoder)
    json.dump(obj=annot_entities, fp=open('../datasets/full/both_anotators.json', 'w'), indent=4, cls=NpEncoder)
