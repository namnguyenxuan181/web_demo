import numpy as np
import json
from abc import ABC, abstractmethod


def load_json(fname):
    with open(fname, encoding='utf-8') as fi:
        return json.load(fi)


class DataIteratorAbstract(ABC):

    @abstractmethod
    def __init__(self, batch_size=64, is_train=False):
        self.batch_size = batch_size
        self.is_train = is_train
        # raise NotImplemented()
        pass

    @abstractmethod
    def process_title(self, title):
        raise NotImplemented()

    @abstractmethod
    def preprocess_data(self, origin_data):
        raise NotImplemented()

    @abstractmethod
    def data(self):
        """
        Create pairs.
        :return:
        """
        raise NotImplemented()

    def __len__(self):
        return int(np.ceil(len(self._data)/self.batch_size))

    def create_batches(self):
        if self.is_train:
            print('Rebuild data.')
            self._data = self.data()

        self.batches = batch(self._data, self.batch_size)

    def init_epoch(self):
        self.create_batches()

    @abstractmethod
    def minibatch2tensor(self, minibatch):
        raise NotImplemented()

    @abstractmethod
    def minibatch2raw(self, minibatch):
        raise NotImplemented()

    def get_batch(self):
        for idx, minibatch in enumerate(self.batches):
            yield self.minibatch2tensor(minibatch)

    def get_batch_analysis(self):
        self.batches = batch(self._data, self.batch_size)
        for idx, minibatch in enumerate(self.batches):
            yield self.minibatch2raw(minibatch), self.minibatch2tensor(minibatch)

    def __iter__(self):
        self.init_epoch()

        for minibatch in self.get_batch():
            yield minibatch

        # gen = self.get_batch()

        # while True:
        #     try:
        #         minibatch = next(gen)
        #         yield minibatch
        #     except StopIteration:
        #         raise StopIteration()
        #     self.init_epoch()
        #     gen = self.get_batch()
        #     minibatch = next(gen)
        #     yield minibatch

    def save(self, save_path):
        import torch
        torch.save(self._data, save_path)


def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch
