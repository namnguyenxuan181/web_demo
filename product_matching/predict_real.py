import os
import pandas as pd
from .measure import InvertIndex, inverted_indices_top_n


def save_results(df, save_dir):
    df.to_csv(os.path.join(save_dir, f'result.tsv'), sep='\t', index=False)


def create_core_products(df):
    core_products = {}
    df_core_product = df[['id', 'title', 'price', 'url']].drop_duplicates()
    # print('total competitor products', len(df_core_product))

    for _, row in df_core_product.iterrows():
        core_products[row.id] = {
            'title': row.title,
            'price': row.price,
            'url': row.url
        }
    return core_products


def create_products(df):
    products = {}
    df_products = df[['pv_sku', 'supplier_product_id', 'supplier_name', 'price', 'url_path']].drop_duplicates()
    # print('total pv products', len(df_products))

    for _, row in df_products.iterrows():
        products[int(row.supplier_product_id)] = {
            'pv_sku': row.pv_sku,
            'supplier_product_id': row.supplier_product_id,
            'pv_title': row.supplier_name,
            'price': row.price,
            'url_path': row.url_path
        }
    return products


def rank(list_product_id: list, products: dict, title: str, core_id, score_function):
    """

    :param core_id:
    :param list_product_id: list candidate PV product ID.
    :param products: products dict to get PV title.
    :param title: anchor title, competitor title.
    :param score_function:
    :return:
    """
    list_columns2 = ['core_id', 'supplier_product_id', 'core_product', 'pv_title', 'score']

    df_results = pd.DataFrame(columns=list_columns2)

    list_pv_title = []

    for product_id in list_product_id:
        supplier_name = products[product_id]['pv_title']
        list_pv_title.append(supplier_name)
        df_results = df_results.append({'core_id': core_id,
                                        'supplier_product_id': product_id,
                                        'core_product': title,
                                        'pv_title': supplier_name},
                                       ignore_index=True)

    scores = score_function(title, list_pv_title)

    df_results['score'] = pd.Series(scores, index=df_results.index)

    df_sorted = df_results.sort_values(by=['score', 'pv_title'], ascending=False)
    return df_sorted


def predict(core_products: dict, products: dict, invert_index: InvertIndex, score_function, save_dir):
    list_columns = ['core_id', 'competitor_title', 'pv_sku', 'pv_title', 'competitor_price', 'pv_price',
                    'competitor_url', 'pv_url', 'score']
    predict_dfs = pd.DataFrame(columns=list_columns)
    for index, (core_id, core_product) in enumerate(sorted(core_products.items())):
        title = core_product['title']
        dict_index_product_id: dict = invert_index.find_products_v2(title)
        list_index_product_id = [v for values in dict_index_product_id.values() for v in values]
        list_product_id_top_n = inverted_indices_top_n(dict_index_product_id)
        if len(list_index_product_id) == 0:
            predict_dfs = predict_dfs.append({
                'core_id': core_id,
                'competitor_title': title,
                'pv_title': ' ',
                'score': -1
            }, ignore_index=True)
            continue
        df_sorted = rank(list_product_id_top_n,
                         products=products,
                         title=title,
                         core_id=core_id,
                         score_function=score_function)
        if index % (len(core_products) // 100) == 0:
            print(len(list_product_id_top_n))
            print(index / len(core_products),
                  len(list_product_id_top_n) / len(products))

        top10 = df_sorted[:10]
        for pro in top10.values:
            if pro[4] == 0:
                continue
            product = products[pro[1]]
            predict_dfs = predict_dfs.append({
                'core_id': core_id,
                'competitor_title': title,
                'pv_sku': product['pv_sku'],
                'pv_title': pro[3],
                'competitor_price': core_product['price'],
                'pv_price': product['price'],
                'competitor_url': core_product['url'],
                'pv_url': product['url_path'],
                'score': pro[4]
            }, ignore_index=True)
            predict_dfs.to_csv(save_dir + '/result.tsv', sep='\t', index=False)
    save_results(predict_dfs, save_dir)
    return predict_dfs


def main(data_dir, date_version, data_kind, model, model_dir, pv_data_kind, output_dir):
    # print('pv_data_kind', pv_data_kind)
    fpath = os.path.join(data_dir, f'{date_version}.{data_kind}.tsv')
    df = pd.read_csv(fpath, sep='\t')
    core_products: dict = create_core_products(df)  # competitors products
    del df

    fpath = os.path.join(data_dir, f'{date_version}.{pv_data_kind}.tsv')
    df = pd.read_csv(fpath, sep='\t')
    products: dict = create_products(df)
    del df

    # print('len phongvu products', len(products))

    inverted_indices = InvertIndex(input_dir=data_dir, date_version=date_version, data_kind=pv_data_kind)
    output_dir = output_dir + '/' + data_kind
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if model == 'triplet':
        from .predictor import get_triplet
        inference = get_triplet(model_dir)
        predict(core_products, products, inverted_indices, score_function=inference.score,
                save_dir=output_dir)
    elif model == 'manhattan':
        from .predictor import get_manhattan
        inference = get_manhattan(model_dir)
        predict(core_products, products, inverted_indices, score_function=inference.score,
                save_dir=output_dir)
    elif model == 'jaro':
        from .jaro_score import score_list_title2
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        predict(core_products, products, inverted_indices, score_function=score_list_title2,
                save_dir=output_dir)
    else:
        raise ValueError


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure model')
    parser.add_argument('--model', type=str, choices=['triplet', 'jaro', 'manhattan'], default='jaro')
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        help='the save_dir of MODEL, needed for AI model.',
                        default=None)
    parser.add_argument('--data-dir', dest='data_dir', type=str, help='input dir', default='datasets/190626')
    parser.add_argument('--date-version', dest='date_version', type=str, help='date version', default='190626')
    parser.add_argument('--data-kind', dest='data_kind', type=str, choices=['laptop', 'monitor', 'mouse', 'keyboard'],
                        default='laptop')
    parser.add_argument('--pv-data-kind', dest='pv_data_kind', type=str, choices=['full_pv_products'],
                        default='full_pv_products')
    parser.add_argument('--output-dir', dest='output_dir', type=str, help='dir to save result file',
                        default='results/jcs_results')

    args = parser.parse_args()
    main(**vars(args))
