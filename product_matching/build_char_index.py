import pandas
import string
import unicodedata

vietnamese_characters = 'aăâbcdđeêghiklmnoôơpqrstuưvxy'


list_fname = [
    'datasets/190626/190626.train.tsv',
    'datasets/190626/190626.val.tsv',
    'datasets/190626/190626.train.tsv',
    'datasets/190626/190626.all.tsv'
]


except_chars = ['\xa0', '\n', '\t', '\u200e']


char_set = set()

for fname in list_fname:
    df = pandas.read_csv(fname, sep='\t')
    for _, row in df.iterrows():
        text = unicodedata.normalize('NFC', row.title + ' ' + row.supplier_name)
        text = text.lower()
        # if '\u200e' in text:
        #     print('\u200e', text)
        # if '\n' in text:
        #     print('\\n', text)
        # if '\xa0' in text:
        #     print(['\xa0'], text, [text])
        for char in except_chars:
            if char in text:
                text = text.replace(char, '')

        char_set = char_set.union(set(text))


char_set = char_set.union(set(string.ascii_lowercase)).union(set(string.digits)).union(set(vietnamese_characters))

char_set = list(sorted(list(char_set)))
print(char_set)
print(len(char_set))

assert not set(vietnamese_characters).difference(char_set)
assert not set(string.ascii_lowercase).difference(char_set)
assert not set(string.digits).difference(char_set)
print('miss punctuation', set(string.punctuation).difference(char_set))

with open('datasets/190626/190626.char_list.txt', 'w', encoding='utf-8') as fo:
    for char in char_set:
        fo.write(f'{char}\n')
