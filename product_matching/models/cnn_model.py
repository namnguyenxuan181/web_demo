import torch
from torch import nn
from . import CharacterEmbedding
from .character_word_models import LstmMultiAtt, filter_sents, make_tokens_att_full
import torch.nn.functional as F
from . import compute_loss_from_att_weights


class CNN(nn.Module):

    def __init__(self, in_channels, kernel_sizes, number_kernel, activation='relu'):
        """
        :param kernel_sizes: list kernel sizes
        :param number_kernel: list number kernels
        :param in_channels: embedding size
        """
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.number_kernel = number_kernel

        self.conv_list = nn.ModuleList([nn.Conv1d(self.in_channels, out_channels=self.number_kernel,
                                                  kernel_size=kernel_size) for kernel_size in self.kernel_sizes])
        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs.transpose(1, 2)
        x = [self.activation(conv(x)) for conv in self.conv_list]
        x = [F.max_pool1d(i, i.shape[2], return_indices=False) for i in x]
        x = torch.cat(x, 1)
        return x.transpose(1, 2)


class CNNBase(nn.Module):
    def __init__(self, opt, num_embedding_chars):
        super(CNNBase, self).__init__()
        self.opt = opt
        self.char_embeddings = CharacterEmbedding(opt['char_embedding_dim'], num_embedding_chars)

        self.word_level = CNN(in_channels=opt['char_embedding_dim'], kernel_sizes=opt['kernel_sizes'],
                              number_kernel=opt['kernel_num'])

        self.sen_level = LstmMultiAtt(opt['kernel_num']*len(opt['kernel_sizes']), opt['lstm_hidden_size'],
                                      opt['lstm_num_layers'],
                                      opt['attention_size'], attention_hops=opt['attention_hops'],
                                      drop_out=opt['drop_out'])

    def forward(self, products, sen_lens, word_lens):
        batch_size, max_word, max_char = products.shape
        flatten_product = products.flatten(end_dim=-2)
        flatten_word_lens: torch.Tensor = word_lens.flatten()

        flatten_product, flatten_word_lens = filter_sents(flatten_product, flatten_word_lens)

        flatten_products_emb: torch.Tensor = self.char_embeddings(flatten_product)
        word_vectors = self.word_level(flatten_products_emb)
        if torch.isnan(word_vectors).sum() > 0:
            print('products', products)
            print('sen_lens', sen_lens)
            print('word_lens', word_lens)
            print('word_vectors', word_vectors)
            raise ValueError('word_vectors have Nan')
        word_vectors = make_tokens_att_full(word_vectors, sen_lens, batch_size, max_word)

        # word_vectors = word_vectors.view(batch_size, max_word, -1)
        assert word_vectors.shape[:2] == (batch_size, max_word)

        sen_vectors, sen_attention_scores = self.sen_level(word_vectors, sen_lens)

        return sen_vectors, (sen_attention_scores,)


class TripletCNN(nn.Module):
    def __init__(self, opt, num_embedding_chars, p=2, custom_loss=False):
        super(TripletCNN, self).__init__()
        self.basemodel = CNNBase(opt, num_embedding_chars)
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
        sentence_vectors, (sen_att_scores,) = self.basemodel(products, sen_lens, word_lens)

        # if self.norm_e:
        #     sentence_vectors = F.normalize(sentence_vectors, p=2.0, dim=-1)

        if not is_predict:
            batch_size = int(products.shape[0] / 3)
            if self.custom_loss:
                custom_loss = compute_loss_from_att_weights(sen_att_scores)
                custom_loss = self.penalty_ratio * torch.stack(custom_loss)
                return torch.split(sentence_vectors, batch_size, dim=0), custom_loss
            return torch.split(sentence_vectors, batch_size, dim=0)
        else:
            batch_size = products.shape[0] - 1
            anchor = sentence_vectors[:1]
            anchor = anchor.repeat((batch_size, 1))
            candidate = sentence_vectors[1:]
            return torch.exp(-torch.norm(anchor - candidate, p=self.p, dim=-1))
