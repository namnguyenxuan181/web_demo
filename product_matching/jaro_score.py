import numpy as np
import pandas as pd
import os

scriptpath = os.path.realpath(__file__)
STOPWORDS = pd.read_csv(scriptpath[:-13]+'stopwords.csv')
STOPWORDS = [i[0] for i in STOPWORDS.values.tolist()]


def pre_process(text):
    text = text.lower().replace('(', ' ').replace(')', ' ').replace('-', ' ').replace(',', ' ').replace('.', ' ')
    text = text.replace('+', ' ').replace('inch', ' ').replace('\'', ' ').replace('\"', ' ').replace(',', ' ').replace(
        '/', ' ').replace('  ', ' ').replace('  ', ' ')
    text = text.replace('  ', ' ').replace('i5 ', 'i5-').replace('i3 ', 'i3-').replace('i7 ', 'i7-')
    text = text.split()
    a = ''
    for t in text:
        if t not in STOPWORDS and len(t) > 1 and 'i3-' not in t and 'i7-' not in t and 'i5-' not in t:
            a += ' ' + t
    if a == '':
        a = ' '.join(text)
    return a.strip()


def get_score(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    m = np.zeros((len_str1, len_str2), dtype=int)  # use to save length of longest common substring
    for i in range(len_str1):
        for j in range(len_str2):
            if str1[i] == str2[j]:
                if i == 0 or j == 0:
                    m[i, j] = 1
                else:
                    m[i, j] = m[i - 1, j - 1] + 1
    lcs = m.max().max()
    return (lcs / len_str1 + lcs / len_str2) / 2


def get_max_score(st, texts, lengh):
    max_score = 0
    for i in range(len(texts)):
        score = (get_score(st, texts[i]))
        if score > max_score:
            max_score = score
    if max_score <= 0.75:
        return 0
    return max_score * len(st) / lengh


def get_score_between_two_title(title1, title2):
    """
    title1: pv title
    title2: match title
    """
    title1 = pre_process(title1)
    title2 = pre_process(title2)
    title1 = title1.split()
    tmp1 = title1 + ['...']
    lengh = len(''.join(title1))
    title2 = title2.split()
    tmp2 = ''.join(title2)
    score = 0
    for i in range(len(title1)):
        if (tmp1[i] + tmp1[i + 1]) in tmp2:
            score += len(tmp1[i] + tmp1[i + 1]) / lengh
        score += get_max_score(title1[i], title2, lengh)
    return score


def get_top_title(name, pv_names, top_n=1):
    """
    find top top_n title similar with  laptop name

    :param string name - laptop name
    :param int top_n - number title want to return

    :return: top n title most similar
    """
    result = {}
    for title in pv_names:
        result[title] = get_score_between_two_title(title, name)
    return [k for k in sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]


def get_all_top_for_product(list_match_name, list_pv_name, top_n=1):
    result = {}
    for name in list_match_name:
        result[name] = get_top_title(name, list_pv_name, top_n)
    return result


def score_list_title2(title1, list_title2):
    out = []
    for title2 in list_title2:
        out.append(get_score_between_two_title(title1, title2))
    return np.array(out)
