from nltk.tokenize import word_tokenize


def text2labels(sentence):
    tokens = word_tokenize(sentence.lower())

    labels = []
    for i, token in enumerate(tokens):
        try:
            if token not in ['.', ',', '?', '!']:
                labels.append('O')
            elif token in ['.', '?', '!']:
                labels[-1] = 'PERIOD'
            elif token == ',':
                labels[-1] = 'COMMA'

        except IndexError:
            raise ValueError(f"Sentence can't start with punctuation {token}")
    return labels


def main():
    text = "Olá, Mundo! Irei compra-los a dinheiro!"
    print(word_tokenize(text))

    print(text2labels(text))


if __name__ == '__main__':
    main()
