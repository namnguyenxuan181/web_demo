import torch
import torch.nn.functional as F
from ..trainers import Trainer, unpack_data, clip_gradient
from ..data_iterator.rank_iterator import RankCharacterWordIterator
from ..models.charword_ranking import TripletCharWordModel
from ..measure import InvertIndex


class TripletCharWordTrainer(Trainer):
    """
    This is idea of Nat, training with only positive.
    Here I try the loss
    """

    def __init__(self, opt, kind):
        # self.loss_margin = 1.0
        # self.loss_p = 2
        # self.loss_p = 1
        self.loss_margin = opt.triplet_margin
        self.loss_p = opt.triplet_norm_p
        super(TripletCharWordTrainer, self).__init__(opt, kind=kind)

    def prepare_data(self, is_train=True):
        # inverted_indices = InvertIndex(input_dir='datasets/190626', date_version='190626', data_kind='all')
        inverted_indices = InvertIndex(input_dir='datasets/190626', date_version='190626', data_kind='full_pv_products')
        if is_train:
            self.train_iter = RankCharacterWordIterator('datasets/190626', '190626', 'train',
                                                        inverted_indices,
                                                        batch_size=self.batch_size,
                                                        # is_train=False)
                                                        is_train=True)
            self.val_iter = RankCharacterWordIterator('datasets/190626', '190626', 'val',
                                                      inverted_indices,
                                                      # batch_size=self.batch_size * 2)
                                                      batch_size=self.batch_size)
        else:
            self.val_iter = RankCharacterWordIterator('datasets/190626', '190626', 'val',
                                                      inverted_indices,
                                                      # batch_size=self.batch_size * 2)
                                                      batch_size=self.batch_size)
            self.test_iter = RankCharacterWordIterator('datasets/190626', '190626', 'test',
                                                       inverted_indices,
                                                       # batch_size=self.batch_size * 2)
                                                       batch_size=self.batch_size)

    def prepare_model(self, opt, is_train=True):
        from ..char_utils import CHAR_LIST_KIND_1

        num_embedding = len(CHAR_LIST_KIND_1)

        self.model = TripletCharWordModel(opt=vars(opt), num_embedding_chars=num_embedding, p=self.loss_p)

    def call_model(self, batch):
        inputs = unpack_data(batch)
        prediction = self.model(*inputs)

        loss = self.loss_fn(prediction, target=None)
        return prediction, None, loss

    def metrics(self, y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return: {'metric_name': score}
        """
        pass

    def loss_fn(self, prediction, target=None):
        """
        Triplet loss.
        :param prediction:
        :param target:
        :return:
        """
        return F.triplet_margin_loss(anchor=prediction[0], positive=prediction[1], negative=prediction[2],
                                     margin=self.loss_margin,
                                     p=self.loss_p)

    def analysis(self, save_dir):
        pass

    def run_epoch_analysis(self, data_iter):
        pass

    def run_epoch(self, data_iter, epoch=None, is_train=False):
        total_epoch_loss = 0

        if not is_train:
            self.model.eval()
        else:
            self.model.train()
            count_backward = 0
            self.optim.zero_grad()

        # with torch.no_grad():
        for step, batch in enumerate(data_iter):

            prediction, target, loss = self.call_model(batch)

            if is_train:
                if torch.isnan(loss).sum() > 0:
                    print('step', step)
                    print('prediction', prediction)
                    print('target', target)
                    raise ValueError("Loss = NaN")
                loss.backward()
                count_backward += 1
                if count_backward == self.count_backward or step == len(data_iter) - 1:
                    # print(step, is_train, len(data_iter), count_backward)

                    clip_gradient(self.model, 1e-1)
                    self.optim.step()
                    count_backward = 0
                    self.optim.zero_grad()

            # num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            total_epoch_loss += loss.item()

            if is_train and step % (len(data_iter) // 5) == 0:
                print('Epoch', epoch, step, len(data_iter), total_epoch_loss / (step + 1), loss.item())

        return total_epoch_loss / len(data_iter), None, None, None

    def evaluate(self):
        """
        Evaluate model.
        :return:
        """
        print('val data:')
        val_loss, val_acc, y_gt, y_prediction = self.run_epoch(self.val_iter)
        print('loss:', val_loss)
        print('test data:')
        test_loss, test_acc, y_gt, y_prediction = self.run_epoch(self.test_iter)
        print('loss:', test_loss)

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()

        print("Start Train")
        for epoch in range(self.num_epoch):
            train_loss, train_acc, _, _ = self.run_epoch(self.train_iter, epoch, is_train=True)
            val_loss, val_acc, y_gt, y_prediction = self.run_epoch(self.val_iter)

            # print(classification_report(y_gt, y_prediction, digits=4, labels=[0, 1],
            #                             target_names=['NEGATIVE', 'POSITIVE']))

            print(f"Epoch: {epoch + 1:d}, loss:{train_loss}, acc:{train_acc}, "
                  f"val_loss:{val_loss}, val_acc {val_acc}")

            self.save_model(epoch, 0., val_loss)

        return self.save_dir


class TripletCharWord2Trainer(TripletCharWordTrainer):
    def __init__(self, opt, kind):
        super().__init__(opt, kind)
        #self.lamda = 0.001
        # self.lamda = 0.2
        self.lamda = 1.0
        print('lamda =', self.lamda)

    def run_epoch(self, data_iter, epoch=None, is_train=False):
        total_epoch_loss = 0

        if not is_train:
            self.model.eval()
        else:
            self.model.train()
            count_backward = 0
            self.optim.zero_grad()

        # with torch.no_grad():
        for step, batch in enumerate(data_iter):

            prediction, target, loss = self.call_model(batch)
            # score_positive = torch.exp(-torch.norm(prediction[0] - prediction[1], 1, dim=-1))
            # positive_loss = F.mse_loss(score_positive, torch.ones(len(prediction[0])).cuda())
            positive_loss = torch.mean(torch.norm(prediction[0] - prediction[1], 1, dim=-1))
            # positive_lamda = self.lamda * positive_loss
            positive_lamda = positive_loss
            if is_train and step % (len(data_iter) // 5) == 0:
                print(f'Epoch:{epoch}, {step}, {len(data_iter)}, triplet={loss.item():.4f}, '
                      f'pos_loss={positive_loss.item():.4f}, pos_loss*lamda={positive_lamda}')

            loss = loss + positive_lamda

            if is_train:
                if torch.isnan(loss).sum() > 0:
                    print('step', step)
                    print('prediction', prediction)
                    print('target', target)
                    raise ValueError("Loss = NaN")
                loss.backward()
                count_backward += 1
                if count_backward == self.count_backward or step == len(data_iter) - 1:
                    # print(step, is_train, len(data_iter), count_backward)

                    clip_gradient(self.model, 1e-1)
                    self.optim.step()
                    count_backward = 0
                    self.optim.zero_grad()

            # num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            total_epoch_loss += loss.item()

            if is_train and step % (len(data_iter) // 5) == 0:
                print('Epoch', epoch, step, len(data_iter), total_epoch_loss / (step + 1), loss.item())

        return total_epoch_loss / len(data_iter), None, None, None

    # def loss_fn(self, prediction, target=None):
    #     """
    #
    #     :param prediction:
    #     :param target:
    #     :return:
    #     """
    #
    #     score_positive = torch.exp(-torch.norm(prediction[0] - prediction[1], 1, dim=-1))
    #     # score_negative = torch.exp(-torch.norm(prediction[0] - prediction[2], 1, dim=-1))
    #
    #     return F.triplet_margin_loss(anchor=prediction[0], positive=prediction[1], negative=prediction[2],
    #                                  margin=self.loss_margin,
    #                                  p=self.loss_p) + F.mse_loss(score_positive, 1)


class TripletCharWord3Trainer(TripletCharWordTrainer):
    """
    Use custom loss for multi attention.
    """
    def __init__(self, opt, kind):
        super(TripletCharWord3Trainer, self).__init__(opt, kind)

    def prepare_model(self, opt, is_train=True):
        from ..char_utils import CHAR_LIST_KIND_1

        num_embedding = len(CHAR_LIST_KIND_1)

        self.model = TripletCharWordModel(opt=vars(opt), num_embedding_chars=num_embedding, p=self.loss_p,
                                          custom_loss=True)

    def call_model(self, batch):
        inputs = unpack_data(batch)
        prediction, custom_loss = self.model(*inputs)

        loss = self.loss_fn(prediction, target=None)
        return prediction, None, (loss, custom_loss)

    def run_epoch(self, data_iter, epoch=None, is_train=False):
        total_epoch_loss = 0

        if not is_train:
            self.model.eval()
        else:
            self.model.train()
            count_backward = 0
            self.optim.zero_grad()

        # with torch.no_grad():
        for step, batch in enumerate(data_iter):
            prediction, target, (loss, custom_loss) = self.call_model(batch)

            if is_train and step % (len(data_iter) // 5) == 0:
                print(f'Epoch:{epoch+1} {step} {len(data_iter)} triplet={loss.item()}, '
                      f'cus_loss={custom_loss.tolist()}')
            loss += custom_loss.sum()

            if is_train:
                if torch.isnan(loss).sum() > 0:
                    print('step', step)
                    print('prediction', prediction)
                    print('target', target)
                    raise ValueError("Loss = NaN")
                loss.backward()
                count_backward += 1
                if count_backward == self.count_backward or step == len(data_iter) - 1:
                    # print(step, is_train, len(data_iter), count_backward)

                    clip_gradient(self.model, 1e-1)
                    self.optim.step()
                    count_backward = 0
                    self.optim.zero_grad()

            # num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            total_epoch_loss += loss.item()

            if is_train and step % (len(data_iter) // 5) == 0:
                print('Epoch', epoch+1, step, len(data_iter), total_epoch_loss / (step + 1), loss.item())

        return total_epoch_loss / len(data_iter), None, None, None
