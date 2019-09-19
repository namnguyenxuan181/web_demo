import torch
import torch.nn.functional as F
from . import Trainer, unpack_data
from ..models.character_word_models import CharacterWordModel, ManhattanCharWordModel
from ..data_iterator.character_word_iterator import CharacterWordIterator
from sklearn.metrics import accuracy_score


class CharWordTrainer(Trainer):
    """
    This is idea of Nat, training with only positive.
    Here I try the loss
    """
    def __init__(self, opt, kind):
        self.labels = [-1, 1]
        self.target_names = ['negative', 'positive']
        self.labels_dict = {k: v for k, v in zip(self.target_names, self.labels)}
        super(CharWordTrainer, self).__init__(opt, kind=kind)

    def prepare_data(self, is_train=True):
        if is_train:
            self.train_iter = CharacterWordIterator('datasets/190626', '190626', 'train', batch_size=self.batch_size,
                                                    is_train=False, only_positive=True, labels=self.labels_dict)
            self.val_iter = CharacterWordIterator('datasets/190626', '190626', 'val', batch_size=self.batch_size * 2,
                                                  only_positive=False, labels=self.labels_dict,
                                                  data_path='datasets/190626/190626.val.pairs.pk')
        else:
            self.test_iter = CharacterWordIterator('datasets/190626', '190626', 'test', batch_size=self.batch_size * 2,
                                                   only_positive=False, labels=self.labels_dict,
                                                   data_path='datasets/190626/190626.test.pairs.pk')

    def prepare_model(self, opt, is_train=True):
        from ..char_utils import CHAR_LIST_KIND_1

        num_embedding = len(CHAR_LIST_KIND_1)

        self.model = CharacterWordModel(opt=vars(opt), num_embedding_chars=num_embedding)

        if is_train:
            if opt.optim == 'adam':
                self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
            elif opt.optim == 'rmsprop':
                self.optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()))

    def call_model(self, batch):
        *inputs, target = unpack_data(batch)
        prediction = self.model(*inputs)

        loss = self.loss_fn(prediction, target.float())
        y_predict = F.cosine_similarity(*prediction).view(target.size()) >= 0
        # print(y_predict, target)
        return y_predict, target, loss

    def metrics(self, y_true, y_pred, *args):
        """

        :param y_true:
        :param y_pred:
        :return: {'metric_name': score}
        """
        return {'accuracy': accuracy_score(y_true, y_pred)}

    def loss_fn(self, prediction, target):
        """
        Cosine loss
        :param prediction:
        :param target:
        :return:
        """
        return F.cosine_embedding_loss(*prediction, target)


class ManhattanCharWordTrainer(CharWordTrainer):
    """
    This is idea of Nat, training with only positive.
    Here I try the loss
    """
    def __init__(self, opt, kind):
        self.labels = [0, 1]
        self.target_names = ['negative', 'positive']
        self.labels_dict = {k: v for k, v in zip(self.target_names, self.labels)}
        super(CharWordTrainer, self).__init__(opt, kind=kind)

    def prepare_data(self, is_train=True):
        if is_train:
            self.train_iter = CharacterWordIterator('datasets/190626', '190626', 'train', batch_size=self.batch_size,
                                                    is_train=True, only_positive=False, labels=self.labels_dict)
            self.val_iter = CharacterWordIterator('datasets/190626', '190626', 'val', batch_size=self.batch_size * 2,
                                                  only_positive=False, labels=self.labels_dict,
                                                  data_path='datasets/190626/190626.val.pairs.pk')
        else:
            self.val_iter = CharacterWordIterator('datasets/190626', '190626', 'val', batch_size=self.batch_size * 2,
                                                  only_positive=False, labels=self.labels_dict,
                                                  data_path='datasets/190626/190626.val.pairs.pk')
            self.test_iter = CharacterWordIterator('datasets/190626', '190626', 'test', batch_size=self.batch_size * 2,
                                                   only_positive=False, labels=self.labels_dict,
                                                   data_path='datasets/190626/190626.test.pairs.pk')

    def prepare_model(self, opt, is_train=True):
        from ..char_utils import CHAR_LIST_KIND_1

        num_embedding = len(CHAR_LIST_KIND_1)

        self.model = ManhattanCharWordModel(opt=vars(opt), num_embedding_chars=num_embedding)

    def call_model(self, batch):
        *inputs, target = unpack_data(batch)
        prediction = self.model(*inputs)

        loss = self.loss_fn(prediction, target.float())
        # print(prediction)
        y_predict = prediction >= 0.5
        # print(y_predict, target)
        return y_predict, target, loss

    def loss_fn(self, prediction, target):
        return F.mse_loss(prediction, target)
