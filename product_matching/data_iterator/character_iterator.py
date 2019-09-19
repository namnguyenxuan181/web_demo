import os
import json
import random
import torch
from ..char_utils import text2char_indices, NEGATIVE, POSITIVE, CHAR_LIST_KIND_0
from . import DataIteratorAbstract, load_json

DATA_LEVEL_0 = 0     # char level, no split title by space.
DATA_LEVEL_1 = 1     # char level, split title by space


class CharacterIterator(DataIteratorAbstract):
    data_level = DATA_LEVEL_0

    def __init__(self, input_dir, date_version, data_kind, batch_size=64, is_train=False):
        super(CharacterIterator, self).__init__(batch_size=batch_size, is_train=is_train)

        f_id_maps = os.path.join(input_dir, f'{date_version}.{data_kind}.id_maps.json')
        f_core_product = os.path.join(input_dir, f'{date_version}.{data_kind}.core_product.json')
        f_products = os.path.join(input_dir, f'{date_version}.{data_kind}.products.json')

        self.id_maps = load_json(f_id_maps)
        self.core_product = self.preprocess_data(load_json(f_core_product))
        self.products = self.preprocess_data(load_json(f_products))
        self._data = self.data()

    def process_title(self, text):
        return text2char_indices(text, kind=self.data_level)

    def preprocess_data(self, origin_data):
        data = {}
        for key, value in origin_data.items():
            data[key] = self.process_title(value['title'])
        return data

    @staticmethod
    def create_pair(supplier_product_id: int, list_positive_id: str, list_negative_id: str):
        """

        :param supplier_product_id:
        :param list_positive_id:
        :param list_negative_id:
        :return: Return examples containing supplier_product_id, positive_id or negative_id and the label.
        """
        data = []
        for positive_id in list_positive_id:
            # print('(supplier_product_id, positive_id), POSITIVE)', (supplier_product_id, positive_id), POSITIVE)
            data.append(((supplier_product_id, positive_id), POSITIVE))

        for negative_id in list_negative_id:
            # print('(supplier_product_id, positive_id), POSITIVE)', (supplier_product_id, negative_id), NEGATIVE)
            data.append(((supplier_product_id, negative_id), NEGATIVE))
        return data

    def data(self):
        data = []
        for category_id in self.id_maps:
            list_supplier_product_id = list(self.id_maps[category_id].keys())
            list_chosen_supplier_product_id = random.choices(list_supplier_product_id,
                                                             k=int(0.8 * len(list_supplier_product_id)))

            for supplier_product_id in list_chosen_supplier_product_id:
                list_core_product_id = list(
                    self.id_maps[category_id][supplier_product_id])
                list_chosen_positive_id = random.choices(list_core_product_id,
                                                         k=int(0.8 * len(list_core_product_id)))

                if len(list_supplier_product_id) > 1:
                    list_negative_product_id = list_supplier_product_id.copy()
                    list_negative_product_id.remove(supplier_product_id)
                    list_negative_id = []
                    for negative_product_id in list_negative_product_id:
                        list_negative_id.extend(self.id_maps[category_id][negative_product_id])
                    list_chosen_negative_id = random.choices(list_negative_id,
                                                             k=int(min(0.8 * len(list_negative_id),
                                                                       # 3 * len(list_chosen_positive_id))))
                                                                       1 * len(list_chosen_positive_id))))
                else:
                    list_chosen_negative_id = []
                data.extend(self.create_pair(supplier_product_id, list_chosen_positive_id, list_chosen_negative_id))
        random.shuffle(data)
        return data

    def _ids2data(self, supplier_product_id, core_product_id):
        # print(supplier_product_id, core_product_id)
        return self.products[supplier_product_id], self.core_product[core_product_id]

    def minibatch2tensor(self, minibatch):
        max_len = None
        data_supplier = []
        data_core = []
        labels = []

        for (supplier_product_id, core_product_id), label in minibatch:
            supplier_product, core_product = self._ids2data(supplier_product_id, core_product_id)
            _current_max_len = max(len(supplier_product), len(core_product))
            if max_len is None or max_len < _current_max_len:
                max_len = _current_max_len

            data_supplier.append(supplier_product)
            data_core.append(core_product)
            labels.append(label)

        products = torch.zeros(2 * len(minibatch), max_len, dtype=torch.long)
        lens = []

        index = 0
        for product in data_supplier:
            products[index][:len(product)] = torch.LongTensor(product)
            index += 1
            lens.append(len(product))
        for product in data_core:
            # print("len(product)", len(product))
            products[index][:len(product)] = torch.LongTensor(product)
            index += 1
            lens.append(len(product))

        # lengths = torch.IntTensor(lens)
        # labels = torch.IntTensor(labels)

        lengths = torch.LongTensor(lens)
        labels = torch.LongTensor(labels)

        return products, lengths, labels


def _check_iterator():
    data_iterator = CharacterIterator('datasets/190626', '190626', 'val', batch_size=1)
    # data_iter = iter(data_iterator)
    # print(next(data_iter))
    for index, _ in enumerate(data_iterator):
        if index == 0:
            a = _
        if index == 1:
            b = _
        # print(_)

    # for index, _ in enumerate(data_iterator):
    #     if index == 0:
    #         b = _

    # print('a', a)
    # print('b', b)

    for batch in (a, b):
        # print(batch)
        product_titles, len_, label = batch
        pv = ''.join([CHAR_LIST_KIND_0[c] for c in product_titles[0].tolist() if c != 0])
        print(pv)
        com = ''.join([CHAR_LIST_KIND_0[c] for c in product_titles[1].tolist() if c != 0])
        print(com)
        print(len_)
        print(label)




if __name__ == '__main__':
    _check_iterator()
