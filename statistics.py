import json


def main():
    annotator1 = json.load(open("annotator1.json", "r"))
    annotator2 = json.load(open("annotator2.json", "r"))
    bert_annotator = json.load(open("bert_annotator.json", "r"))
