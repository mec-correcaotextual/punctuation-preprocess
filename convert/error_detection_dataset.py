import json
import os
import pathlib
from collections import defaultdict
from typing import Literal

import srsly

from convert.util import fix_punctuation, read_data
from utils import text2labels, find_token_span, get_gold_token, NpEncoder
from utils.preprocess import preprocess_text, fix_break_lines
import pandas as pd


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
        new_text = '\n'.join(new_sts_text)

        entity = {'text': new_sts_text, "title": title, 'text_id': text_id}

        annots = []
        for annotation in group_annottion['label'].values:

            for k, ann_span in enumerate(annotation):
                start_char, end_char = ann_span[1], ann_span[1] + 2
                # Descobrir o porquê há multiplas pontuações no texto do aluno e corrigir isso 'esta podre.?,
                start_char, end_char = get_gold_token(text, start_char, end_char)
                ann_span[0] = start_char
                ann_span[1] = end_char
                annotation[k] = ann_span
                annots.append(ann_span)

        entity['raw_text'] = text
        entity['raw_text_id'] = text_id
        entity['text'] = new_text
        entity['annotations'] = annots

        annotator_entities.append(entity)

    return annotator_entities


if __name__ == '__main__':
    annot_entities = convert_annotations('../annotations/')

    json.dump(obj=annot_entities, fp=open('../datasets/annotator.json', 'w'), indent=4, cls=NpEncoder)
