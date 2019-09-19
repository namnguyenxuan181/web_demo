import os
import json
import time
import pandas as pd
from .core_nlp import normalize, isnotpunct

# THRESHOLD_INVERTED_TOP_N = 50
THRESHOLD_INVERTED_TOP_N = 20


def load_json(fpath):
    with open(fpath, encoding='utf-8') as fi:
        return json.load(fi)


def create_core_products(df):
    core_products = {}
    df_core_product = df[['id', 'title', 'supplier_product_id']].drop_duplicates()
    print('len df_core_product', len(df_core_product))

    for _, row in df_core_product.iterrows():
        core_products[row.id] = {
            'title': row.title,
            'supplier_product_id': row.supplier_product_id
        }
    return core_products


def create_products(df):
    products = {}
    # df_core_product = df[['id', 'title', 'price', 'url']].drop_duplicates()
    df_products = df[['supplier_product_id', 'supplier_name']].drop_duplicates()
    print('len df_products', len(df_products))

    for _, row in df_products.iterrows():
        products[int(row.supplier_product_id)] = {
            'title': row.supplier_name,
        }
    return products


class InvertIndex:
    """
    This invertIndex for phong vu products.
    """

    def __init__(self, input_dir='datasets/190626', date_version='190626', data_kind='all'):
        fname = os.path.join(input_dir, f'{date_version}.{data_kind}.tsv')

        df = pd.read_csv(fname, sep='\t')

        df['supplier_name'] = df['supplier_name'].apply(normalize)
        # df['title'] = df['title'].apply(normalize)

        df_products = df[['supplier_product_id', 'supplier_name']].drop_duplicates()
        self.invert_dict = self.create_invert(df_products)
        del df_products

    @staticmethod
    def create_invert(df_products):
        invert_dict = {}

        # len_products = len(df_products)

        word_punct = set()

        for _, row in df_products.iterrows():
            splited_text = row.supplier_name.lower().split(' ')
            for word in splited_text:
                if isnotpunct(word):
                    if word not in invert_dict:
                        invert_dict[word] = {row.supplier_product_id}
                    else:
                        invert_dict[word].add(row.supplier_product_id)
                else:
                    word_punct.add(word)

        print('inverted_punctuations', word_punct)
        print('len_inverted', len(invert_dict))

        return invert_dict

    @staticmethod
    def text2list(text):
        text = normalize(text).lower()
        words = text.split(' ')
        words = [w for w in words if isnotpunct(w)]
        return words

    def find_products(self, title):

        words = self.text2list(title)

        results = set()
        for word in words:
            if word in self.invert_dict:
                results.update(self.invert_dict[word])

        return results

    def find_products_v2(self, title):
        words = self.text2list(title)

        results = {}
        for word in words:
            if word in self.invert_dict:
                for product_id in self.invert_dict[word]:
                    if product_id in results:
                        results[product_id] += 1
                    else:
                        results[product_id] = 1

        results_sort = {}
        for k, v in results.items():
            if v in results_sort:
                results_sort[v].append(k)
            else:
                results_sort[v] = [k]

        return results_sort


def inverted_indices_top_n(dict_index_product_id, top_n=THRESHOLD_INVERTED_TOP_N):
    list_product_id = []
    for score_count in sorted(dict_index_product_id, reverse=True):
        # print('score_count', score_count)
        list_ = dict_index_product_id[score_count]
        list_product_id.extend(list_)
        if len(list_product_id) >= top_n:
            break

    return list_product_id


def rank(list_product_id: list, products: dict, title: str, core_id, supplier_product_id: int, score_function):
    """

    :param list_product_id: list candidate PV product ID.
    :param products: products dict to get PV title.
    :param title: anchor title, competitor title.
    :param supplier_product_id: True ID
    :param score_function:
    :return:
    """
    # TODO: need return rank and top 10 + theirs score.
    list_columns2 = ['core_id', 'supplier_product_id', 'core_product', 'supplier_name', 'score', 'true_false']

    df_results = pd.DataFrame(columns=list_columns2)

    list_pv_title = []

    for product_id in list_product_id:
        supplier_name = products[product_id]['title']
        list_pv_title.append(supplier_name)
        df_results = df_results.append({'core_id': core_id,
                                        'supplier_product_id': product_id,
                                        # 'score': get_score_between_two_title(title, supplier_name),
                                        'core_product': title,
                                        'supplier_name': supplier_name,
                                        'true_false': supplier_product_id == product_id},
                                       ignore_index=True)

    scores = score_function(title, list_pv_title)

    df_results['score'] = pd.Series(scores, index=df_results.index)

    index = df_results.index[df_results['supplier_product_id'] == supplier_product_id]
    if len(index) == 0:
        return 10000000
    assert len(index) == 1

    true_score = df_results.loc[index].score.values[0]
    list_indices = df_results.score >= true_score
    len_ = list_indices.sum()

    if len_ > 10:
        df_sorted = df_results[list_indices].sort_values(by=['score', 'supplier_name'], ascending=False)
    else:
        df_sorted = df_results.sort_values(by=['score', 'supplier_name'], ascending=False)
        if len_ > 5:  # thuoc top 6-10
            df_sorted = df_sorted[:15]
        elif len_ > 1:  # thuoc top 2-5
            df_sorted = df_sorted[:10]
        else:  # thuoc top 1
            df_sorted = df_sorted[:5]
    return len_, df_sorted


def _add_to_df_result(df_dict, key, core_id, supplier_product_id, core_product, supplier_name):
    df_dict[key] = df_dict[key].append({
        'core_id': core_id,
        'supplier_product_id': supplier_product_id,
        'core_product': core_product,
        'supplier_name': supplier_name
    }, ignore_index=True)


def _add_to_df_score_result(df_dict, key, other_df: pd.DataFrame):
    df_dict[key] = pd.concat([df_dict[key], other_df], ignore_index=True, sort=False)


def evaluate(core_products: dict, products: dict, invert_index: InvertIndex, score_function, save_dir):
    begin_time = time.time()

    count_true = {
        'inverted': 0,
        'inverted_top_n': 0,
        'top_1': 0,
        'top_5': 0,
        'top_10': 0
    }

    list_columns = ['core_id', 'supplier_product_id', 'core_product', 'supplier_name']

    fail_dfs = {
        'inverted': pd.DataFrame(columns=list_columns),
        'inverted_top_n': pd.DataFrame(columns=list_columns),
        'top_1': pd.DataFrame(columns=list_columns),
        'top_5': pd.DataFrame(columns=list_columns),
        'top_10': pd.DataFrame(columns=list_columns)
    }

    success_dfs = {
        'top_1': pd.DataFrame(columns=list_columns),
        'top_5': pd.DataFrame(columns=list_columns),
        'top_10': pd.DataFrame(columns=list_columns)
    }

    list_columns2 = ['core_id', 'supplier_product_id', 'core_product', 'supplier_name', 'score', 'true_false']

    fail_score_dfs = {
        'top_1': pd.DataFrame(columns=list_columns2),
        'top_5': pd.DataFrame(columns=list_columns2),
        'top_10': pd.DataFrame(columns=list_columns2)
    }

    success_score_dfs = {
        'top_1': pd.DataFrame(columns=list_columns2),
        'top_5': pd.DataFrame(columns=list_columns2),
        'top_10': pd.DataFrame(columns=list_columns2)
    }

    for index, (core_id, core_product) in enumerate(sorted(core_products.items())):
        title = core_product['title']  # title of competitor's product.
        supplier_product_id = core_product['supplier_product_id']

        dict_index_product_id: dict = invert_index.find_products_v2(title)
        list_index_product_id = [v for values in dict_index_product_id.values() for v in values]

        if len(list_index_product_id) == 0:
            _add_to_df_result(fail_dfs, 'inverted', core_id, supplier_product_id, title,
                              products[supplier_product_id]['title'])
            continue

        if supplier_product_id in list_index_product_id:
            count_true['inverted'] += 1
        else:
            _add_to_df_result(fail_dfs, 'inverted', core_id, supplier_product_id, title,
                              products[supplier_product_id]['title'])
            continue

        # Get top N by inverted index.
        list_product_id_top_n = inverted_indices_top_n(dict_index_product_id)

        if supplier_product_id in list_product_id_top_n:
            count_true['inverted_top_n'] += 1
        else:
            _add_to_df_result(fail_dfs, 'inverted_top_n', core_id, supplier_product_id, title,
                              products[supplier_product_id]['title'])
            continue

        if index % (len(core_products) // 100) == 0:
            print(len(list_product_id_top_n))
            print(index / len(core_products), len(list_product_id_top_n) / len(list_index_product_id),
                  len(list_product_id_top_n) / len(products))

        xep_hang, df_score = rank(list_product_id_top_n,
                                  products=products,
                                  title=title,
                                  core_id=core_id,
                                  supplier_product_id=supplier_product_id,
                                  score_function=score_function)

        if xep_hang <= 10:
            count_true['top_10'] += 1
            if xep_hang <= 5:
                count_true['top_5'] += 1
                if xep_hang <= 1:
                    count_true['top_1'] += 1
                    _add_to_df_result(success_dfs, 'top_1', core_id, supplier_product_id, title,
                                      products[supplier_product_id]['title'])
                    _add_to_df_score_result(success_score_dfs, 'top_1', df_score)

                else:
                    _add_to_df_result(fail_dfs, 'top_1', core_id, supplier_product_id, title,
                                      products[supplier_product_id]['title'])
                    _add_to_df_score_result(fail_score_dfs, 'top_1', df_score)

                    _add_to_df_result(success_dfs, 'top_5', core_id, supplier_product_id, title,
                                      products[supplier_product_id]['title'])
                    _add_to_df_score_result(success_score_dfs, 'top_5', df_score)

            else:
                _add_to_df_result(fail_dfs, 'top_5', core_id, supplier_product_id, title,
                                  products[supplier_product_id]['title'])
                _add_to_df_score_result(fail_score_dfs, 'top_5', df_score)

                _add_to_df_result(success_dfs, 'top_10', core_id, supplier_product_id, title,
                                  products[supplier_product_id]['title'])
                _add_to_df_score_result(success_score_dfs, 'top_10', df_score)

        else:
            _add_to_df_result(fail_dfs, 'top_10', core_id, supplier_product_id, title,
                              products[supplier_product_id]['title'])
            _add_to_df_score_result(fail_score_dfs, 'top_10', df_score)

    end_time = time.time()

    print("time per len core products", (end_time - begin_time) / len(core_products))
    print('index', count_true['inverted'] / len(core_products))
    print('inverted_index THRESHOLD_INVERTED_TOP_N={}'.format(THRESHOLD_INVERTED_TOP_N),
          count_true['inverted_top_n'] / len(core_products))
    print('top1', count_true['top_1'] / len(core_products))
    print('top5', count_true['top_5'] / len(core_products))
    print('top10', count_true['top_10'] / len(core_products))

    save_results(success_dfs, fail_dfs, success_score_dfs, fail_score_dfs, len(core_products),
                 end_time - begin_time, count_true, save_dir)


def save_results(success_dfs: dict, fail_dfs: dict,
                 success_score_dfs: dict, fail_score_dfs: dict,
                 len_data: int,
                 total_time: float,
                 count_true, save_dir):
    for df_name, df, in success_dfs.items():
        df.to_csv(os.path.join(save_dir, f'success.{df_name}.tsv'), sep='\t', index=False)

    for df_name, df in fail_dfs.items():  # type: str, pd.DataFrame
        df.to_csv(os.path.join(save_dir, f'fail.{df_name}.tsv'), sep='\t', index=False)

    for df_name, df in success_score_dfs.items():  # type: str, pd.DataFrame
        df.to_csv(os.path.join(save_dir, f'success.score.{df_name}.tsv'), sep='\t', index=False)

    for df_name, df in fail_score_dfs.items():  # type: str, pd.DataFrame
        df.to_csv(os.path.join(save_dir, f'fail.score.{df_name}.tsv'), sep='\t', index=False)

    with open(os.path.join(save_dir, 'results.txt'), 'w', encoding='utf-8') as fo:
        fo.write(f'time/len_data: {total_time / len_data}\n')
        for key in ['inverted', 'inverted_top_n', 'top_1', 'top_5', 'top_10']:
            value = count_true[key]
            fo.write(f'{key}: {value / len_data}\n')


def main(data_dir, date_version, data_kind, model, model_dir, pv_data_kind):
    print('pv_data_kind', pv_data_kind)
    fpath = os.path.join(data_dir, f'{date_version}.{data_kind}.tsv')
    df = pd.read_csv(fpath, sep='\t')
    core_products: dict = create_core_products(df)  # competitors products
    del df
    print('len competitor products', len(core_products))

    # fpath = os.path.join(data_dir, f'{date_version}.all.tsv')
    fpath = os.path.join(data_dir, f'{date_version}.{pv_data_kind}.tsv')
    df = pd.read_csv(fpath, sep='\t')
    products: dict = create_products(df)
    del df
    print('len phongvu products', len(products))

    # inverted_indices = InvertIndex(input_dir=data_dir, date_version=date_version, data_kind='all')
    inverted_indices = InvertIndex(input_dir=data_dir, date_version=date_version, data_kind=pv_data_kind)

    if model == 'triplet':
        from .predictor import get_triplet
        inference = get_triplet(model_dir)
        evaluate(core_products, products, inverted_indices, score_function=inference.score,
                 save_dir=model_dir)
    elif model == 'cnn':
        from .predictor import get_cnn
        inference = get_cnn(model_dir)
        evaluate(core_products, products, inverted_indices, score_function=inference.score,
                 save_dir=model_dir)
    elif model == 'manhattan':
        from .predictor import get_manhattan
        inference = get_manhattan(model_dir)
        evaluate(core_products, products, inverted_indices, score_function=inference.score,
                 save_dir=model_dir)
    elif model == 'jaro':
        from .jaro_score import score_list_title2
        # save_dir = 'results/nam_algorithm_results'
        # save_dir = 'results/nam_algorithm_results2'
        save_dir = 'results/nam_algorithm_results3'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        evaluate(core_products, products, inverted_indices, score_function=score_list_title2,
                 save_dir=save_dir)
    else:
        raise ValueError


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure model')
    parser.add_argument('--model', type=str, choices=['triplet', 'jaro', 'manhattan', 'cnn'], default='jaro')
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        help='the save_dir of MODEL, needed for AI model.',
                        default=None)
    # parser.add_argument('--save-dir', dest='save_dir', type=str,
    #                     help='where we save analysis',
    #                     default='nam_algorithm_results'
    #                     )
    parser.add_argument('--data-dir', dest='data_dir', type=str, help='input dir', default='datasets/190626')
    parser.add_argument('--date-version', dest='date_version', type=str, help='date version', default='190626')
    parser.add_argument('--data-kind', dest='data_kind', type=str, choices=['train', 'val', 'test', 'all'],
                        default='test')
    parser.add_argument('--pv-data-kind', dest='pv_data_kind', type=str, choices=['all', 'full_pv_products'],
                        default='all')

    args = parser.parse_args()
    main(**vars(args))
