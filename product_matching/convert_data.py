import os
import json
import pandas
import unicodedata


def convert_to_id_maps(out_dir, date_version, data_kind, df):
    """

    :param out_dir:
    :param date_version:
    :param data_kind:
    :param df:
    :return:
    """
    fname = os.path.join(out_dir, f'{date_version}.{data_kind}.id_maps.json')
    print(f'saving to {fname}')

    list_standard_supplier_category_id = df.standard_supplier_category_id.unique()

    id_maps = {int(i): {} for i in list_standard_supplier_category_id}

    for _, row in df.iterrows():
        if row.supplier_product_id not in id_maps[row.standard_supplier_category_id]:
            id_maps[int(row.standard_supplier_category_id)][int(row.supplier_product_id)] = [row.id]
        else:
            id_maps[int(row.standard_supplier_category_id)][int(row.supplier_product_id)].append(row.id)

    with open(fname, 'w', encoding='utf-8') as fo:
        json.dump(id_maps, fo, ensure_ascii=False)


def convert_to_core_product(out_dir, date_version, data_kind, df):
    """

    :param out_dir:
    :param date_version:
    :param data_kind:
    :param df:
    :return:
    """
    fname = os.path.join(out_dir, f'{date_version}.{data_kind}.core_product.json')
    print(f'saving to {fname}')

    core_product = {}
    # df_core_product = df[['id', 'title', 'price', 'url']].drop_duplicates()
    df_core_product = df[['id', 'title']].drop_duplicates()
    print('len df_core_product', len(df_core_product))

    for _, row in df_core_product.iterrows():
        core_product[row.id] = {
            'title': unicodedata.normalize('NFC', row.title),
        }

    with open(fname, 'w', encoding='utf-8') as fo:
        json.dump(core_product, fo, ensure_ascii=False)


def convert_to_products(out_dir, date_version, data_kind, df):
    """

    :param out_dir:
    :param date_version:
    :param data_kind:
    :param df:
    :return:
    """
    fname = os.path.join(out_dir, f'{date_version}.{data_kind}.products.json')
    print(f'saving to {fname}')

    products = {}
    # df_core_product = df[['id', 'title', 'price', 'url']].drop_duplicates()
    df_products = df[['supplier_product_id', 'supplier_name']].drop_duplicates()
    print('len df_products', len(df_products))

    for _, row in df_products.iterrows():
        products[int(row.supplier_product_id)] = {
            'title': unicodedata.normalize('NFC', row.supplier_name),
        }

    with open(fname, 'w', encoding='utf-8') as fo:
        json.dump(products, fo, ensure_ascii=False)


def convert_data(input_dir, date_version='190626', data_kind='val'):
    """

    :param date_version:
    :param data_kind:
    :return:
    """
    fname = os.path.join(input_dir, f'{date_version}.{data_kind}.tsv')

    df = pandas.read_csv(fname, sep='\t')
    print(df.count())
    print(df.columns)
    out_dir = input_dir
    convert_to_id_maps(out_dir, date_version, data_kind, df)
    convert_to_core_product(out_dir, date_version, data_kind, df)
    convert_to_products(out_dir, date_version, data_kind, df)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert fact data to json')

    parser.add_argument('--input-dir', dest='input_dir', type=str, help='input dir', default='datasets/190626')
    parser.add_argument('--date-version', dest='date_version', type=str, help='date version', default='190626')
    parser.add_argument('--data-kind', dest='data_kind', type=str, choices=['train', 'val', 'test', 'all'],
                        default='val')

    args = parser.parse_args()
    convert_data(**vars(args))
