import re
import unicodedata


EXCEPT_CHARS = ['\xa0', '\n', '\t', '\u200e', '\x1d']

# REGEX = re.compile(r"""([\w]+|\S)""", re.UNICODE)
REGEX = re.compile(r"""(\d+([\.,_]\d+)+|[\w]+|\S)""", re.UNICODE)


def clean_text(text):
    for char in EXCEPT_CHARS:
        if char in text:
            text = text.replace(char, '')
    return text


def normalize(text, raise_error=True):
    text = unicodedata.normalize("NFC", text)
    text = clean_text(text)
    # tokenized_text = " ".join(REGEX.findall(text))
    normolized_text = " ".join([t[0] for t in REGEX.findall(text)])

    if raise_error:
        if normolized_text.replace(' ', '') != text.replace(' ', ''):
            print([text], [normolized_text])
            raise ValueError('Invalid char.')
    return normolized_text


NOPUNCT = re.compile(r"""(\d+([\.,_]\d+)+|[\w]+)""", re.UNICODE)


def isnotpunct(word):
    return bool(NOPUNCT.fullmatch(word))
