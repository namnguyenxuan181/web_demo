import unicodedata
import os

NEGATIVE = 0
POSITIVE = 1

EXCEPT_CHARS = ['\xa0', '\n', '\t', '\u200e']       # '§', '#', '~'


def _load_chars_list(kind=0):
    assert kind in [0, 1]
    file_path = os.path.realpath(__file__)
    file_path = os.path.dirname(os.path.dirname(file_path))
    file_name = os.path.join(file_path, 'data/190626.char_list.txt')
    with open(file_name, 'r', encoding='utf-8') as fi:
        char_list = fi.read()
        if kind == 0:
            char_list = list(char_list.replace('\n', ''))
            # assert len(char_list) == 140
            assert len(char_list) == 139
        elif kind == 1:
            char_list = list(char_list.replace('\n', '').replace(' ', ''))
            # assert len(char_list) == 139
            assert len(char_list) == 138
        else:
            raise ValueError("kind={} not be expected.".format(kind))

    # add PADDING.
    char_list.insert(0, '__PAD__')
    return char_list


CHAR_LIST_KIND_0 = _load_chars_list(0)
CHAR_LIST_KIND_1 = _load_chars_list(1)


def _remove_except_chars(text):
    for char in EXCEPT_CHARS:
        if char in text:
            text = text.replace(char, '')
    return text


def text2char_indices(text, kind):
    text = unicodedata.normalize("NFC", text)
    if kind == 0:
        text = text.lower()
        text = _remove_except_chars(text)

        list_id = [CHAR_LIST_KIND_0.index(c) for c in text]
    elif kind == 1:
        text = text.lower()
        # text = _remove_except_chars(text)
        out_text = ""
        for c in text:
            if c in CHAR_LIST_KIND_1:
                out_text += c
        text = out_text

        list_id = []
        for word in text.split():
            word_id = []
            for c in word:
                try:
                    word_id.append(CHAR_LIST_KIND_1.index(c))
                except ValueError as e:
                    # print(e)
                    print(text)
                    # raise e
            list_id.append(word_id)

    else:
        raise ValueError("kind={} not be expected.".format(kind))

    return list_id
