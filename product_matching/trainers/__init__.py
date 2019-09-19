import os
import torch
from sklearn.metrics import accuracy_score, classification_report
from product_matching.utils import save_args, load_args, get_best_model
from abc import ABC, abstractmethod
import datetime


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def unpack_data(batch):
    if torch.cuda.is_available():
        out = []
        for x in batch:
            out.append(x.cuda())
        return out
    else:
        return batch


class Trainer(ABC):
    def __init__(self, opt, kind='train'):
        assert kind in ['train', 'eval', 'predict']
        self.batch_size = opt.batch_size
        self.model = None

        if kind == 'train':
            self.root_save_dir = opt.save_dir
            self.num_epoch = opt.epoch
            self.count_backward = opt.count_backward
            print('real batch size', self.count_backward * self.batch_size)
            self.train_iter = None
            self.val_iter = None
            self.save_dir = None

            self.optim = None

            self.train_init(opt)
        elif kind == 'eval':
            self.test_iter = None

            self.eval_init(opt)

        elif kind == 'predict':
            self.predict_init(opt)

    @abstractmethod
    def prepare_data(self, is_train=True):
        """
        Create iter for train or for test
        :param is_train:
        :return:
        """
        raise NotImplementedError

    def train_init(self, opt):
        self.prepare_data(is_train=True)
        self.prepare_model(opt)
        self.prepare_optimize(opt)
        self.prepare_save_dir(opt)

    def eval_init(self, opt):
        self.prepare_data(is_train=False)
        self.prepare_model(opt, is_train=False)

    def predict_init(self, opt):
        self.prepare_model(opt, is_train=False)

    @abstractmethod
    def prepare_model(self, opt, is_train=True):
        raise NotImplementedError

    def prepare_optimize(self, opt):
        if opt.optim == 'adam':
            self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        elif opt.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()))

    def prepare_save_dir(self, opt):
        save_dir = os.path.join(self.root_save_dir, f"{datetime.datetime.now().strftime('%y%m%d%H%M%S'):}-{opt.model}")
        # prepare save_dir
        assert not os.path.exists(save_dir)
        print("SAVE_DIR", save_dir)
        os.makedirs(save_dir)

        print('args', vars(opt))
        save_args(opt, save_dir)
        self.save_dir = save_dir

    @abstractmethod
    def loss_fn(self, prediction, target=None):
        raise NotImplementedError

    @abstractmethod
    def metrics(self, y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return: {'metric_name': score}
        """
        # return accuracy_score(y_true, y_pred)
        raise NotImplementedError

    @abstractmethod
    def call_model(self, batch):
        """

        :param batch:
        :return: prediction, target, loss
        """

        # *inputs, target = unpack_data(batch)
        # prediction = self.model(*inputs)
        #
        # loss = self.loss_fn(prediction, target)
        # prediction = torch.max(prediction, 1)[1].view(target.size())
        raise NotImplementedError

    def run_epoch(self, data_iter, epoch=None, is_train=False):
        total_epoch_loss = 0

        y_pred = []
        y_true = []

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

            # y_pred.extend(torch.max(prediction, 1)[1].view(target.size()).tolist())
            y_pred.extend(prediction.tolist())
            y_true.extend(target.tolist())

            metrics = self.metrics(y_true, y_pred)

            if is_train:
                self.print_log_step(step, len(data_iter), epoch, loss, metrics)

        return total_epoch_loss / len(data_iter), accuracy_score(y_true, y_pred), y_true, y_pred

    def run_epoch_analysis(self, data_iter):
        """
        It help analysis
        :param data_iter:
        :return:
        """
        total_epoch_loss = 0

        data = []
        y_pred = []
        y_true = []

        self.model.eval()
        # with torch.no_grad():
        for step, batch in enumerate(data_iter.get_batch_analysis()):
            data_raw, batch = batch
            prediction, target, loss = self.call_model(batch)
            total_epoch_loss += loss.item()
            y_pred.extend(prediction.tolist())
            y_true.extend(target.tolist())
            data.extend(data_raw)

        return total_epoch_loss / len(data_iter), accuracy_score(y_true, y_pred), y_true, y_pred, data

    @staticmethod
    def print_log_step(step, max_step, epoch, loss, metrics):
        if (step + 1) % (max_step // 5) == 0:

            text = f'Epoch: {epoch + 1: 4d}, Idx: {step + 1}, Training Loss: {loss.item():.4f}, '

            for key, value in metrics.items():
                text += f'{key}: {value: .4f}%'

            print(text)

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

            self.save_model(epoch, val_acc, val_loss)

        return self.save_dir

    def evaluate(self):
        """
        Evaluate model.
        :return:
        """
        print('val data:')
        test_loss, test_acc, y_gt, y_prediction = self.run_epoch(self.val_iter)
        print(classification_report(y_gt, y_prediction, digits=4, labels=self.labels,
                                    target_names=self.target_names))
        print('test data:')
        test_loss, test_acc, y_gt, y_prediction = self.run_epoch(self.test_iter)
        print(classification_report(y_gt, y_prediction, digits=4, labels=self.labels,
                                    target_names=self.target_names))
        return self

    def analysis(self, save_dir):
        """
        Evaluate model.
        :return:
        """
        print('analysis test data:')
        test_loss, test_acc, y_gt, y_prediction, data = self.run_epoch_analysis(self.test_iter)
        print(classification_report(y_gt, y_prediction, digits=4, labels=self.labels,
                                    target_names=self.target_names))

        with open(os.path.join(save_dir, 'analysis.tsv'), 'w') as fi:
            for data_raw, y_true, y_pred in zip(data, y_gt, y_prediction):
                fi.write(f'{data_raw[0][0]}\t{data_raw[0][1]}\t{y_true}\t{y_pred}\t{y_true==y_pred}\n')

        with open(os.path.join(save_dir, 'analysis.positive.true.tsv'), 'w') as fi:
            for data_raw, y_true, y_pred in zip(data, y_gt, y_prediction):
                if y_true == y_pred and y_true == 1:
                    fi.write(f'{data_raw[0][0]}\t{data_raw[0][1]}\t{y_true}\t{y_pred}\t{y_true==y_pred}\n')

        with open(os.path.join(save_dir, 'analysis.positive.false.tsv'), 'w') as fi:
            for data_raw, y_true, y_pred in zip(data, y_gt, y_prediction):
                if y_true != y_pred and y_true == 1:
                    fi.write(f'{data_raw[0][0]}\t{data_raw[0][1]}\t{y_true}\t{y_pred}\t{y_true==y_pred}\n')

        with open(os.path.join(save_dir, 'analysis.negative.true.tsv'), 'w') as fi:
            for data_raw, y_true, y_pred in zip(data, y_gt, y_prediction):
                if y_true == y_pred and y_true == 0:
                    fi.write(f'{data_raw[0][0]}\t{data_raw[0][1]}\t{y_true}\t{y_pred}\t{y_true==y_pred}\n')

        with open(os.path.join(save_dir, 'analysis.negative.false.tsv'), 'w') as fi:
            for data_raw, y_true, y_pred in zip(data, y_gt, y_prediction):
                if y_true != y_pred and y_true == 0:
                    fi.write(f'{data_raw[0][0]}\t{data_raw[0][1]}\t{y_true}\t{y_pred}\t{y_true==y_pred}\n')

    def save_model(self, epoch, val_acc, val_loss):
        save_path = f'{self.save_dir}/epoch:{epoch + 1:04d}_acc:{val_acc:4f}_loss:{val_loss:4f}.pth'

        print(f'Saving model to {save_path}')
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        state_dict = torch.load(save_path,  map_location={'cuda:0': 'cpu'})
        print("state_dict.keys", state_dict.keys())

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.load_state_dict(state_dict)

    @classmethod
    def load_from_saved_dir(cls, save_dir, kind='eval'):
        assert kind in ['eval', 'predict']
        opt = load_args(save_dir)
        print('args', vars(opt))

        self = cls(opt, kind=kind)

        model_path = get_best_model(save_dir, 2, reverse=False)
        self.load_model(model_path)

        return self
