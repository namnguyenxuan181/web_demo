import os
from . import DataIteratorAbstract, load_json
from ..char_utils import text2char_indices, NEGATIVE, POSITIVE, CHAR_LIST_KIND_1
import random
import torch


class CharacterWordIterator(DataIteratorAbstract):
    data_level = 1

    def __init__(self, input_dir, date_version, data_kind, batch_size=64, is_train=False, only_positive=False,
                 labels=None, data_path=None):
        """

        :param input_dir:
        :param date_version:
        :param data_kind:
        :param batch_size:
        :param is_train:
        :param only_positive:
        :param labels: default: `{'negative': 0, 'positive': 1}`
        """
        super(CharacterWordIterator, self).__init__()
        self.is_train = is_train
        self.only_positive = only_positive
        self.labels = labels or {'negative': 0, 'positive': 1}

        f_id_maps = os.path.join(input_dir, f'{date_version}.{data_kind}.id_maps.json')
        f_core_product = os.path.join(input_dir, f'{date_version}.{data_kind}.core_product.json')
        f_products = os.path.join(input_dir, f'{date_version}.{data_kind}.products.json')

        self.id_maps = load_json(f_id_maps)
        self.core_product_raw = load_json(f_core_product)
        self.core_product = self.preprocess_data(self.core_product_raw)
        self.products_raw = load_json(f_products)
        self.products = self.preprocess_data(self.products_raw)
        self.batch_size = batch_size
        # if self.is_train:
        if data_path is None:
            self._data = self.data()
        else:
            print(f'Re-load {data_kind} from {data_path}.')
            assert is_train is False
            self._data = torch.load(data_path)

    def process_title(self, text):
        return text2char_indices(text, kind=self.data_level)

    def preprocess_data(self, origin_data):
        data = {}
        for key, value in origin_data.items():
            data[key] = self.process_title(value['title'])
        return data

    def create_pair(self, supplier_product_id: int, list_positive_id: str, list_negative_id: str):
        """

        :param supplier_product_id:
        :param list_positive_id:
        :param list_negative_id:
        :return: Return examples containing supplier_product_id, positive_id or negative_id and the label.
        """
        data = []
        for positive_id in list_positive_id:
            data.append(((supplier_product_id, positive_id), 'positive'))

        for negative_id in list_negative_id:
            data.append(((supplier_product_id, negative_id), 'negative'))
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

                if not self.only_positive and len(list_supplier_product_id) > 1:
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
        max_len_sen = None
        max_len_word = None
        data_supplier = []
        data_core = []
        labels = []

        # find max len
        for (supplier_product_id, core_product_id), label in minibatch:
            supplier_product, core_product = self._ids2data(supplier_product_id, core_product_id)
            _current_max_len_sen = max(len(supplier_product), len(core_product))
            _current_max_len_word = max([len(w) for sen in (supplier_product, core_product) for w in sen])
            if max_len_sen is None or max_len_sen < _current_max_len_sen:
                max_len_sen = _current_max_len_sen

            if max_len_word is None or max_len_word < _current_max_len_word:
                max_len_word = _current_max_len_word

            data_supplier.append(supplier_product)
            data_core.append(core_product)
            label = self.labels[label]
            labels.append(label)

        products = torch.zeros(2 * len(minibatch), max_len_sen, max_len_word, dtype=torch.long)
        sen_lens = torch.zeros(2 * len(minibatch), dtype=torch.long)
        word_lens = torch.zeros(2 * len(minibatch), max_len_sen, dtype=torch.long)

        # id to tensor
        index = 0
        for product in data_supplier:
            for word_id, word in enumerate(product):
                products[index][word_id][:len(word)] = torch.LongTensor(word)
                word_lens[index][word_id] = len(word)

            sen_lens[index] = len(product)
            index += 1

        for product in data_core:
            for word_id, word in enumerate(product):
                products[index][word_id][:len(word)] = torch.LongTensor(word)
                word_lens[index][word_id] = len(word)

            sen_lens[index] = len(product)

            index += 1

        labels = torch.LongTensor(labels)

        return products, sen_lens, word_lens, labels

    def minibatch2raw(self, minibatch):
        data = []

        for (supplier_product_id, core_product_id), label in minibatch:
            data.append(((self.products_raw[supplier_product_id]['title'],
                          self.core_product_raw[core_product_id]['title']), label))

        return data


def _check_iterator(only_positive):
    data_iterator = CharacterWordIterator('datasets/190626', '190626', 'val', batch_size=1, only_positive=only_positive)

    for index, _ in enumerate(data_iterator):
        if index == 0:
            a = _
        if index == 1:
            b = _

    def word(list_indices):
        return ''.join([CHAR_LIST_KIND_1[c] for c in list_indices if c != 0])

    for batch in (a, b):
        # print(batch)
        product_titles, len_sen, len_word, label = batch
        pv = ' '.join([word(w) for w in product_titles[0].tolist()])
        print(pv)
        com = ' '.join([word(w) for w in product_titles[1].tolist()])

        print(com)
        print(len_sen)
        print(len_word)
        print(label)

    for index, (product_titles, len_sen, len_word, label) in enumerate(data_iterator):
        if index >= 10:
            break
        print(label)


def create_val_test():
    labels = [0, 1]
    target_names = ['negative', 'positive']
    labels_dict = {k: v for k, v in zip(target_names, labels)}
    batch_size = 64
    val_iter = CharacterWordIterator('datasets/190626', '190626', 'val', batch_size=batch_size * 2,
                                     only_positive=False, labels=labels_dict)
    val_iter.save('datasets/190626/190626.val.pairs.pk')
    test_iter = CharacterWordIterator('datasets/190626', '190626', 'val', batch_size=batch_size * 2,
                                      only_positive=False, labels=labels_dict)
    test_iter.save('datasets/190626/190626.test.pairs.pk')


if __name__ == '__main__':
    # _check_iterator(only_positive=True)
    create_val_test()
