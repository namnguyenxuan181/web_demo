import torch
from torch import nn
import torch.nn.functional as F
from .character_word_models import BaseModel
from . import compute_custom_loss


class TripletCharWordModel(nn.Module):
    def __init__(self, opt, num_embedding_chars, p=2, custom_loss=False):
        super(TripletCharWordModel, self).__init__()
        self.basemodel = BaseModel(opt, num_embedding_chars)
        self.p = p
        self.norm_e = opt['norm_e']
        self.custom_loss = custom_loss
        if self.custom_loss:
            self.penalty_ratio = torch.Tensor(opt['penalty_ratio'])
            if torch.cuda.is_available():
                self.penalty_ratio = self.penalty_ratio.cuda()
            print('penalty_ratio', self.penalty_ratio)
        print('norm_p =', self.p)
        print('norm_e =', self.norm_e)

    def forward(self, products, sen_lens, word_lens, is_predict=False):
        """

        :param products:
        :param sen_lens:
        :param word_lens:
        :param is_predict:
        :return:
        """
        sentence_vectors, (word_att_scores, sen_att_scores) = self.basemodel(products, sen_lens, word_lens)

        # if self.norm_e:
        #     sentence_vectors = F.normalize(sentence_vectors, p=2.0, dim=-1)

        if not is_predict:
            batch_size = int(products.shape[0] / 3)
            if self.custom_loss:
                custom_loss = compute_custom_loss(word_att_scores, sen_att_scores, sen_lens)
                custom_loss = self.penalty_ratio * torch.stack(custom_loss)
                return torch.split(sentence_vectors, batch_size, dim=0), custom_loss
            return torch.split(sentence_vectors, batch_size, dim=0)
        else:
            batch_size = products.shape[0] - 1
            anchor = sentence_vectors[:1]
            anchor = anchor.repeat((batch_size, 1))
            candidate = sentence_vectors[1:]
            return torch.exp(-torch.norm(anchor - candidate, p=self.p, dim=-1))
