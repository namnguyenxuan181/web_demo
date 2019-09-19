import torch
from . import Trainer, unpack_data
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class CharacterTrainer(Trainer):

    def __init__(self, opt, is_train):
        self.labels = [0, 1]
        self.target_names = ['negative', 'positive']
        super(CharacterTrainer, self).__init__(opt, is_train)

    def prepare_data(self, is_train=True):
        from ..data_iterator.character_iterator import CharacterIterator

        if is_train:
            self.train_iter = CharacterIterator('datasets/190626', '190626', 'train', batch_size=self.batch_size,
                                                is_train=False)
            self.val_iter = CharacterIterator('datasets/190626', '190626', 'val', batch_size=self.batch_size * 2)
        else:
            self.test_iter = CharacterIterator('datasets/190626', '190626', 'test', batch_size=self.batch_size * 2)

    def prepare_model(self, opt, is_train=True):
        from ..char_utils import CHAR_LIST_KIND_0

        num_embedding = len(CHAR_LIST_KIND_0)

        from ..models.character_models import CharacterModel
        self.model = CharacterModel(opt=vars(opt), num_embedding=num_embedding, output_size=2)

    def loss_fn(self, prediction, target):
        return F.cross_entropy(prediction, target)

    def call_model(self, batch):
        """

        :param batch:
        :return: prediction, target, loss
        """

        *inputs, target = unpack_data(batch)
        prediction = self.model(*inputs)

        loss = self.loss_fn(prediction, target)
        prediction = torch.max(prediction, 1)[1].view(target.size())
        return prediction, target, loss

    def metrics(self, y_true, y_pred, *args):
        """

        :param y_true:
        :param y_pred:
        :return: {'metric_name': score}
        """
        return {'accuracy': accuracy_score(y_true, y_pred)}
        # return accuracy_score(y_true, y_pred)
