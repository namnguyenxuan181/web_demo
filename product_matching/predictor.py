import numpy as np
import torch
from .trainers.char_word_trainer import ManhattanCharWordTrainer, CharacterWordIterator
from .trainers.charword_ranking import TripletCharWordTrainer, RankCharacterWordIterator
from .char_utils import text2char_indices


class ManhattanInference:
    def __init__(self, save_dir):
        print(f'Load model from {save_dir}')
        self.trainer = ManhattanCharWordTrainer.load_from_saved_dir(save_dir, kind='predict')
        # self.trainer.evaluate()
        self.trainer.model.eval()

    def batch2tensor(self, minibatch):
        max_len_sen = None
        max_len_word = None
        data_a = []
        data_b = []

        # find max len
        for text_a, text_b in minibatch:
            try:
                sen_a = text2char_indices(text_a, kind=CharacterWordIterator.data_level)
                sen_b = text2char_indices(text_b, kind=CharacterWordIterator.data_level)
            except ValueError as e:
                print(text_a)
                print(text_b)
                raise e

            _current_max_len_sen = max(len(sen_a), len(sen_b))
            _current_max_len_word = max([len(w) for sen in (sen_a, sen_b) for w in sen])
            if max_len_sen is None or max_len_sen < _current_max_len_sen:
                max_len_sen = _current_max_len_sen

            if max_len_word is None or max_len_word < _current_max_len_word:
                max_len_word = _current_max_len_word

            data_a.append(sen_a)
            data_b.append(sen_b)

        products = torch.zeros(len(minibatch) + 1, max_len_sen, max_len_word, dtype=torch.long)
        sen_lens = torch.zeros(len(minibatch) + 1, dtype=torch.long)
        word_lens = torch.zeros(len(minibatch) + 1, max_len_sen, dtype=torch.long)

        # id to tensor
        index = 0
        for product in data_a:
            for word_id, word in enumerate(product):
                products[index][word_id][:len(word)] = torch.LongTensor(word)
                word_lens[index][word_id] = len(word)

            sen_lens[index] = len(product)
            index += 1
            break   # need only a sen_a

        for product in data_b:
            for word_id, word in enumerate(product):
                products[index][word_id][:len(word)] = torch.LongTensor(word)
                word_lens[index][word_id] = len(word)

            sen_lens[index] = len(product)

            index += 1

        if torch.cuda.is_available():
            return products.cuda(), sen_lens.cuda(), word_lens.cuda()
        return products, sen_lens, word_lens

    def predict(self, text_a, text_b):
        """products, sen_lens, word_lens

        :param text_a: PV product
        :param text_b: Competitor product
        :return:
        """
        # products, sen_lens, word_lens = self.text2tensor(text_a, text_b)
        # prediction = self.trainer.model(products, sen_lens, word_lens)
        # return prediction >= 0.5, prediction
        pass

    def data2batches(self, text_a, list_text_b, batch_size):
        batches = []
        batch = []
        for text_b in list_text_b:
            batch.append((text_a, text_b))

            if len(batch) == batch_size:
                batches.append(batch)
                batch = []

        if batch:
            batches.append(batch)
        return batches

    def score(self, text_a, list_text_b,
              # batch_size=128,
              batch_size=128,
             # top_n=10
             ):

        batches = self.data2batches(text_a, list_text_b, batch_size)

        out = []
        for batch in batches:
            prediction = self.trainer.model(*self.batch2tensor(batch))
            out.append(prediction)

        out = torch.cat(out, dim=0)
        # out[out < 0.5] = 0
        return out.cpu().detach().numpy()
        # label = out >= 0.5
        #
        # sorted_tensor, indices = torch.sort(out, descending=True)
        #
        # return label, indices[0], sorted_tensor[0], indices[:top_n], sorted_tensor[:top_n]


class TripletInference(ManhattanInference):
    def __init__(self, save_dir):
        super(ManhattanInference, self).__init__()
        print(f'Load model from {save_dir}')
        self.trainer = TripletCharWordTrainer.load_from_saved_dir(save_dir, kind='predict')
        # self.trainer.evaluate()
        self.trainer.model.eval()

    def score(self, text_a, list_text_b, batch_size=128,
             # top_n=10
             ):
        batches = self.data2batches(text_a, list_text_b, batch_size)

        out = []
        for batch in batches:
            prediction = self.trainer.model(*self.batch2tensor(batch), is_predict=True)
            out.append(prediction)

        out = torch.cat(out, dim=0)
        return out.cpu().detach().numpy()


class CnnInference(TripletInference):
    def __init__(self, save_dir):
        super(ManhattanInference, self).__init__()
        print(f'Load model from {save_dir}')
        from .trainers.cnn_trainer import CNNTrainer
        self.trainer = CNNTrainer.load_from_saved_dir(save_dir, kind='predict')
        # self.trainer.evaluate()
        self.trainer.model.eval()


def get_manhattan(save_dir):
    # return ManhattanInference(save_dir='saved_dir/190718162340-manhattan-charword')
    return ManhattanInference(save_dir=save_dir)


def get_triplet(save_dir):
    # p=2, nhưng chưa training tới 30 epoch.
    # return TripletInference(save_dir='saved_dir_rank/190719122930-triplet-charword')

    # p=1, đã training tới 30 epochs.
    # return TripletInference(save_dir='saved_dir_rank/190719200817-triplet-charword')

    return TripletInference(save_dir=save_dir)


def get_cnn(save_dir):
    return CnnInference(save_dir=save_dir)
