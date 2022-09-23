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
        if ann_text_list[-1] not in ['.', ',', ';', ':', '!', '?']:
            ann_text_list.append('.')
            shift += 1
        elif ann_text_list[-1] in other_punctuations:
            ann_text_list[-1] = '.'
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

                # Adiciona espaço após a pontuação se necessário
                if end_char + 1 >= len(ann_text_list):
                    ann_text_list.append('.')  # Adiciona ponto final
                    break
                if ann_text_list[i] in [' ', '\n', '\t']:
                    shift -= 1
                else:
                    ann_text_list.insert(i, ' ')

                    shift += 1
                break

    return ann_text_list, shift
