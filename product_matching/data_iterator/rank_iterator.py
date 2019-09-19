import os
from . import DataIteratorAbstract, load_json
from ..char_utils import text2char_indices, NEGATIVE, POSITIVE, CHAR_LIST_KIND_1
import random
import torch
from ..measure import inverted_indices_top_n, create_products
import pandas as pd


class RankCharacterWordIterator(DataIteratorAbstract):
    data_level = 1

    def __init__(self, input_dir, date_version, data_kind, inverted_indices, batch_size=64, is_train=False,
                 # negative_candidate=None,
                 data_path=None):
        """

        :param input_dir:
        :param date_version:
        :param data_kind:
        :param batch_size:
        :param is_train:
        :param only_positive:
        """
        super(RankCharacterWordIterator, self).__init__()
        self.is_train = is_train

        self.inverted_indices = inverted_indices
        self.inverted_top_n = 50

        f_id_maps = os.path.join(input_dir, f'{date_version}.{data_kind}.id_maps.json')
        f_core_product = os.path.join(input_dir, f'{date_version}.{data_kind}.core_product.json')
        # f_products = os.path.join(input_dir, f'{date_version}.all.products.json')  # use all pv products.
        f_products = os.path.join(input_dir, f'{date_version}.full_pv_products.tsv')  # use all pv products.

        self.core_product_raw = load_json(f_core_product)
        self.core_product = self.preprocess_data(self.core_product_raw)
        # self.products_raw = load_json(f_products)
        df_products = pd.read_csv(f_products, sep='\t')
        self.products_raw = create_products(df_products)
        self.products_raw = {str(id_): value for id_, value in self.products_raw.items()}
        self.products = self.preprocess_data(self.products_raw)

        # if negative_candidate is None:
        self.negative_candidates = self.build_negative_candidates(f_id_maps)
        # else:
        #     self.negative_candidates = load_json(negative_candidate)

        self.batch_size = batch_size

        if data_path is None:
            self._data = self.data()
        else:
            print(f'Re-load {data_kind} from {data_path}.')
            assert is_train is False
            self._data = torch.load(data_path)

    def build_negative_candidates(self, f_id_maps):
        id_maps: dict = load_json(f_id_maps)  # this map pv's product with many competitor's products.

        out = {}
        for _, list_pv_product_id in id_maps.items():
            for pv_product_id in list_pv_product_id:
                for competitor_product_id in list_pv_product_id[pv_product_id]:
                    assert competitor_product_id not in out
                    out[competitor_product_id] = {
                        'positive': pv_product_id,
                        'negative': None
                    }
                    title = self.core_product_raw[competitor_product_id]['title']
                    negative_list = self.inverted_indices.find_products_v2(title)
                    negative_list = inverted_indices_top_n(negative_list, top_n=self.inverted_top_n)
                    negative_list = [str(_) for _ in negative_list]
                    if pv_product_id in negative_list:
                        negative_list.remove(pv_product_id)
                    out[competitor_product_id]['negative'] = negative_list
        return out

    def process_title(self, text):
        return text2char_indices(text, kind=self.data_level)

    def preprocess_data(self, origin_data):
        data = {}
        for key, value in origin_data.items():
            data[key] = self.process_title(value['title'])
        return data

    def create_triplet(self, anchor: str, positive_id: int, list_negative_id: list):
        """
        :param achor:
        :param positive_id:
        :param list_negative_id:
        :return: return triplet.
        """
        data = []

        for negative_id in list_negative_id:
            data.append((anchor, positive_id, negative_id))

        return data

    def data(self):
        data = []

        list_anchor = list(self.negative_candidates)
        print('len anchor', len(list_anchor))
        # list_anchor = random.choices(list_anchor, k=int(0.8 * len(list_anchor)))
        print('len anchor reduce', len(list_anchor))

        for competitor_id in list_anchor:
            positive_id = self.negative_candidates[competitor_id]['positive']
            list_negative_id = self.negative_candidates[competitor_id]['negative']
            # list_negative_id = random.choices(list_negative_id, k=min(int(0.8 * len(list_negative_id)), 10))
            list_negative_id = random.choices(list_negative_id, k=min(int(0.8 * len(list_negative_id)), 25))
            data.extend(self.create_triplet(competitor_id, positive_id, list_negative_id))
        print('len data', len(data))
        random.shuffle(data)
        return data

    def _ids2data(self, competitor_id, positive_pv_id, negative_pv_id):
        # print(competitor_id, positive_pv_id, negative_pv_id)
        return self.core_product[competitor_id], self.products[positive_pv_id], self.products[negative_pv_id]

    def minibatch2tensor(self, minibatch):
        max_len_sen = None
        max_len_word = None
        data_anchor = []
        data_positive = []
        data_negative = []

        # find max len
        for competitor_id, positive_pv_id, negative_pv_id in minibatch:
            triplet = self._ids2data(competitor_id, positive_pv_id, negative_pv_id)
            _current_max_len_sen = max([len(_) for _ in triplet])
            _current_max_len_word = max([len(w) for sen in triplet for w in sen])
            if max_len_sen is None or max_len_sen < _current_max_len_sen:
                max_len_sen = _current_max_len_sen

            if max_len_word is None or max_len_word < _current_max_len_word:
                max_len_word = _current_max_len_word

            data_anchor.append(triplet[0])
            data_positive.append(triplet[1])
            data_negative.append(triplet[2])

        triplets = torch.zeros(3 * len(minibatch), max_len_sen, max_len_word, dtype=torch.long)
        sen_lens = torch.zeros(3 * len(minibatch), dtype=torch.long)
        word_lens = torch.zeros(3 * len(minibatch), max_len_sen, dtype=torch.long)

        # id to tensor
        index = 0
        for data in [data_anchor, data_positive, data_negative]:
            for product in data:
                for word_id, word in enumerate(product):
                    triplets[index][word_id][:len(word)] = torch.LongTensor(word)
                    word_lens[index][word_id] = len(word)

                sen_lens[index] = len(product)
                index += 1

        return triplets, sen_lens, word_lens

    def minibatch2raw(self, minibatch):
        data = []

        for (supplier_product_id, core_product_id), label in minibatch:
            data.append(((self.products_raw[supplier_product_id]['title'],
                          self.core_product_raw[core_product_id]['title']), label))

        return data


def _check_iterator():
    data_iterator = RankCharacterWordIterator('datasets/190626', '190626', 'val', batch_size=1)

    def word(list_indices):
        return ''.join([CHAR_LIST_KIND_1[c] for c in list_indices if c != 0])

    for index, batch in enumerate(data_iterator):

        # print(batch)
        triplets, len_sen, len_word = batch
        for sen in triplets:
            pv = ' '.join([word(w) for w in sen.tolist()])
            print(pv)
        print(len_sen)
        print(len_word)

        if index >= 10:
            break


# def create_val_test():
#     batch_size = 64
#     val_iter = RankCharacterWordIterator('datasets/190626', '190626', 'val', batch_size=batch_size * 2,
#                                      only_positive=False)
#     val_iter.save('datasets/190626/190626.val.pairs.pk')
#     test_iter = RankCharacterWordIterator('datasets/190626', '190626', 'val', batch_size=batch_size * 2,
#                                       only_positive=False)
#     test_iter.save('datasets/190626/190626.test.pairs.pk')


if __name__ == '__main__':
    _check_iterator()
    # create_val_test()
