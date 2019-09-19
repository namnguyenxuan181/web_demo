import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LstmLayer, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            h_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        return h_0, c_0

    def forward(self, sequence_input, seq_lengths):
        """
        :param sequence_input:
        :param seq_lengths:
        :return: [batch_size, max_word, 2*rnn_size]
        """
        batch_size, max_len = sequence_input.shape[:2]

        sorted_seq_lengths, argsort_lengths = seq_lengths.sort(0, descending=True)

        # invert_argsort_lengths = torch.LongTensor(argsort_lengths.shape).fill_(0).to(self.device)
        if torch.cuda.is_available():
            invert_argsort_lengths = torch.LongTensor(argsort_lengths.shape).fill_(0).cuda()
        else:
            invert_argsort_lengths = torch.LongTensor(argsort_lengths.shape).fill_(0)
        for i, v in enumerate(argsort_lengths):
            invert_argsort_lengths[v.data] = i

        sorted_input = sequence_input[argsort_lengths]
        sorted_input = pack_padded_sequence(sorted_input, sorted_seq_lengths, batch_first=True)

        lstm_outputs, (final_hidden_state, final_cell_state) = self.lstm(sorted_input, self.init_hidden(batch_size))

        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        lstm_outputs = lstm_outputs[invert_argsort_lengths]

        assert not lstm_outputs.shape[1] < max_len

        return lstm_outputs, (final_hidden_state, final_cell_state)


class MultiAttentionLayer(nn.Module):

    def __init__(self, input_size, attention_size, attention_hops=1):
        super(MultiAttentionLayer, self).__init__()

        self.attention_hops = attention_hops
        self.linear_first = nn.Linear(input_size, attention_size, bias=False)
        self.linear_second = nn.Linear(attention_size, attention_hops, bias=False)

    def forward(self, seq_input, lengths):
        """
        :param seq_input: [bsize, max_len, ]
        :param lengths:
        :return:
        """
        batch_size, max_len = seq_input.shape[:-1]
        x = torch.tanh(self.linear_first(seq_input))  # [bsize, max_len, att_size]
        x = self.linear_second(x)  # [bsize, max_len, att_hops]
        attention = x.transpose(1, 2)  # [bsize, att_hops, max_len]

        if torch.cuda.is_available():
            mask = torch.arange(max_len).cuda().expand(batch_size, max_len) < lengths.unsqueeze(1)
        else:
            mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).repeat(1, self.attention_hops, 1)
        attention[~mask] = -float('inf')

        attention = attention.softmax(-1)  # [bsize, att_hops, max_len]
        seq_embedding = attention @ seq_input  # [bsize, att_hops, emb_size]

        avg_seq_embedding = seq_embedding.mean(1)

        return avg_seq_embedding, attention


class CharacterEmbedding(nn.Module):
    def __init__(self, char_embedding_dim, num_embedding):
        super(CharacterEmbedding, self).__init__()
        self.embeddings = nn.Embedding(num_embedding, char_embedding_dim)

    def forward(self, x):
        return self.embeddings(x)


def compute_loss_from_att_weights(att_weights):
    tranpose_w = torch.transpose(att_weights, 1, 2)
    loss = (att_weights @ tranpose_w - torch.eye(att_weights.shape[1]).cuda()).norm(dim=(1, 2))
    return loss


def compute_loss_word_level(att_weights0, document_lengths):
    loss_word_levels = compute_loss_from_att_weights(att_weights0)

    list_word_levels = []

    pre_index = 0
    for length in document_lengths:
        list_word_levels.append(loss_word_levels[pre_index:pre_index + length].mean().unsqueeze(0))
        pre_index += length

    sum_loss_word_levels = torch.cat(list_word_levels)
    return sum_loss_word_levels


def compute_custom_loss(att_weights0, att_weights1, document_lengths):
    return compute_loss_word_level(att_weights0, document_lengths).mean(), \
           compute_loss_from_att_weights(att_weights1).mean()
