import pandas as pd
import logging
import os
import numpy as np
from .measure import inverted_indices_top_n, InvertIndex
from .predict_real import create_core_products, create_products, rank
from .core_nlp import normalize, isnotpunct
from .jaro_score import score_list_title2

_logger = logging.getLogger(__name__)


class InvertIndexV2(InvertIndex):
    def __init__(self, products):
        super(InvertIndex, self).__init__()
        df_products = pd.DataFrame(products[['supplier_product_id', 'supplier_name']])
        # df_products['supplier_name'] = df_products['supplier_name'].apply(normalize, args=(False,))
        df_products['supplier_name'] = df_products['supplier_name'].apply(lambda x: normalize(x, raise_error=False))
        self.invert_dict = self.create_invert(df_products)
        del df_products

    @staticmethod
    def text2list(text):
        text = normalize(text, raise_error=False).lower()
        words = text.split(' ')
        words = [w for w in words if isnotpunct(w)]
        return words


class ProductMatching:
    def __init__(self, products, model_path='data/best_model'):
        from .predictor import get_triplet
        file_path = os.path.realpath(__file__)
        file_path = os.path.dirname(os.path.dirname(file_path))
        model_path = os.path.join(file_path, model_path)
        self.list_model = {'AI': None, 'LCS': score_list_title2}
        self.model = None
        if model_path is not None:
            self.model = get_triplet(model_path)
            self.list_model['AI'] = self.model.score

        self.df_products = products
        self.inverted = InvertIndexV2(self.df_products)

    def get_top_10(self, title, kind_model='AI'):
        """

        :param title: title want to get top 10
        :param kind_model: AI for ai model, LCS for lcs method, combine foe use 2 model
        :return: top 10 pv product
        """
        if (kind_model == 'AI' or kind_model == 'combine') and self.model is None:
            logging.error('do not find AI model, please make sure import model path.')
            return
        list_columns = ['pv_sku', 'pv_title', 'pv_price',
                        'pv_url', 'score']
        predict_dfs = pd.DataFrame(columns=list_columns)
        products = create_products(self.df_products)
        dict_index_product_id: dict = self.inverted.find_products_v2(title)
        list_index_product_id = [v for values in dict_index_product_id.values() for v in values]
        list_product_id_top_n = inverted_indices_top_n(dict_index_product_id)
        if len(list_index_product_id) == 0:
            return predict_dfs

        if kind_model == 'AI' or kind_model == 'LCS':
            # get top 10 product with  kind model
            df_sorted = rank(list_product_id_top_n,
                             products=products,
                             title=title,
                             core_id=None,
                             score_function=self.list_model[kind_model])
            top10 = df_sorted[:10]
            for pro in top10.values:
                product = products[pro[1]]

                # add triplet result
                predict_dfs = predict_dfs.append({
                    'pv_sku': product['pv_sku'],
                    'pv_title': pro[3],
                    'pv_price': product['price'],
                    'pv_url': product['url_path'],
                    'score': pro[4]
                }, ignore_index=True)

        else:  # get 5 results of AI and 5 results of lcs and sort them: odd is AI result
            # get top 5 product with triplet model
            df_sorted = rank(list_product_id_top_n,
                             products=products,
                             title=title,
                             core_id=None,
                             score_function=self.model.score)
            top5_triplet = df_sorted[:5]

            # get top 5 product with LCS algorithm
            df_sorted = rank(list_product_id_top_n,
                             products=products,
                             title=title,
                             core_id=None,
                             score_function=score_list_title2)
            top5_lcs = df_sorted[:5]

            for pro_triplet, pro_lcs in zip(top5_triplet.values, top5_lcs.values):
                product_triplet = products[pro_triplet[1]]
                product_lcs = products[pro_lcs[1]]

                # add triplet result
                predict_dfs = predict_dfs.append({
                    'pv_sku': product_triplet['pv_sku'],
                    'pv_title': pro_triplet[3],
                    'pv_price': product_triplet['price'],
                    'pv_url': product_triplet['url_path'],
                    # 'score': pro_triplet[4]
                }, ignore_index=True)

                # add lcs result
                if pro_lcs[4] == 0:
                    continue
                predict_dfs = predict_dfs.append({
                    'competitor_title': title,
                    'pv_sku': product_lcs['pv_sku'],
                    'pv_title': pro_lcs[3],
                    'pv_price': product_lcs['price'],
                    'pv_url': product_lcs['url_path'],
                    # 'score': pro_lcs[4]
                }, ignore_index=True)
        predict_dfs = predict_dfs.drop_duplicates()
        return predict_dfs

    def create_df(self, core_products):
        list_columns = ['core_id', 'competitor_title', 'pv_sku', 'supplier_product_id', 'pv_title', 'competitor_price', 'pv_price',
                        'competitor_url', 'pv_url', 'score']

        predict_dfs = pd.DataFrame(columns=list_columns)
        predict_dfs = predict_dfs.astype({'competitor_price': 'int', 'competitor_title': 'str', 'competitor_url': 'str',
                                          'core_id': 'str', 'pv_price': 'int', 'pv_sku': 'str', 'score': 'float',
                                          'pv_title': 'str', 'pv_url': 'str', 'supplier_product_id': 'int'})
        core_products = create_core_products(core_products)
        products = create_products(self.df_products)
        return predict_dfs, core_products, products

    def find_candidate(self, title):
        dict_index_product_id: dict = self.inverted.find_products_v2(title)
        list_product_id_top_n = inverted_indices_top_n(dict_index_product_id)
        return list_product_id_top_n

    @staticmethod
    def add_result(predict_dfs, core_id, title, product, pro, core_product, add_score=True):
        """
        :param predict_dfs:
        :param core_id:
        :param title:
        :param product:
        :param pro:
        :param core_product:
        :param add_score:
        :return:
        """

        dict_obj = {
            'core_id': core_id,
            'competitor_title': title,
            'pv_sku': product['pv_sku'],
            'pv_title': pro[3],
            'competitor_price': core_product['price'],
            'pv_price': product['price'],
            'competitor_url': core_product['url'],
            'pv_url': product['url_path'],
        }

        if add_score:
            dict_obj['score'] = pro[4]

        return predict_dfs.append(dict_obj, ignore_index=True)

    def rank_top_10(self, core_id, core_product, products, score_function):
        title = core_product['title']
        list_product_id_top_n = self.find_candidate(title)
        if len(list_product_id_top_n) == 0:
            return
        # get top 10 product with triplet model
        top10 = rank(list_product_id_top_n,
                     products=products,
                     title=title,
                     core_id=core_id,
                     score_function=score_function)[:10]

        list_url_path = [products[pv.supplier_product_id]['url_path'] for _, pv in top10.iterrows()]
        list_price = [products[pv.supplier_product_id]['price'] for _, pv in top10.iterrows()]
        list_sku = [products[pv.supplier_product_id]['pv_sku'] for _, pv in top10.iterrows()]

        # top10['core_id'] = core_id
        top10['pv_url'] = list_url_path
        top10['pv_price'] = list_price
        top10['pv_sku'] = list_sku
        top10['competitor_url'] = core_product['url']
        top10['competitor_price'] = core_product['price']
        # top10['competitor_title'] = title
        top10 = top10.replace({np.nan: None})
        top10 = top10.rename(columns={'core_product': 'competitor_title'})
        return top10

    def predict_m_kind(self, core_products, kind_model='AI'):
        """
        Using to make predict for all competitor product
        :param kind_model:
        :param core_products: competitor products
        :return: DataFrame: content pair of competitor product and pv product
        """

        score_function = self.list_model[kind_model]
        if kind_model == 'AI' and self.model is None:
            logging.error('do not find AI model, please make sure import model path.')
            return
        predict_dfs, core_products, products = self.create_df(core_products)

        for index, (core_id, core_product) in enumerate(sorted(core_products.items())):
            top10 = self.rank_top_10(core_id, core_product, products, score_function)

            if top10 is not None:
                predict_dfs = predict_dfs.append(top10, ignore_index=True)
        return predict_dfs

    def predict_combine(self, core_products):
        """
        use to make predict for all competitor product
        :param core_products: competitor products
        :return: DataFrame: content pair of competitor product and pv product
        """
        if self.model is None:
            logging.error('do not find AI model, please make sure import model path.')
            return

        predict_dfs, core_products, products = self.create_df(core_products)
        for index, (core_id, core_product) in enumerate(sorted(core_products.items())):
            title = core_product['title']
            list_product_id_top_n = self.find_candidate(title)
            if len(list_product_id_top_n) == 0:
                continue

            # get top product with triplet model
            top_triplet = rank(list_product_id_top_n,
                               products=products,
                               title=title,
                               core_id=core_id,
                               score_function=self.model.score)

            # get top product with LCS algorithm
            top_lcs = rank(list_product_id_top_n,
                           products=products,
                           title=title,
                           core_id=core_id,
                           score_function=score_list_title2)
            for pro_triplet, pro_lcs in zip(top_triplet.values, top_lcs.values):
                product_triplet = products[pro_triplet[1]]
                product_lcs = products[pro_lcs[1]]

                # add triplet result
                predict_dfs = self.add_result(predict_dfs, core_id, title,
                                              product_triplet, pro_triplet, core_product, False)
                if pro_lcs[4] == 0:
                    continue
                predict_dfs = self.add_result(predict_dfs, core_id, title, product_lcs, pro_lcs, core_product, False)
                # predict_dfs = predict_dfs.drop_duplicates()
                if len(predict_dfs) >= 10:
                    break

        # predict_dfs = predict_dfs.drop_duplicates()
        return predict_dfs

    def predict(self, core_products, kind_model='combine'):
        """
        :param core_products:
        :param kind_model:
        :return:
        """
        if kind_model in ['AI', 'LCS']:
            result = self.predict_m_kind(core_products, kind_model)
        elif kind_model == 'combine':
            result = self.predict_combine(core_products)
        else:
            raise ValueError('kind_model={} is a valid value.'.format(kind_model))
        return result
