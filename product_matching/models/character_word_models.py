import torch
from torch import nn
from . import LstmLayer, MultiAttentionLayer, CharacterEmbedding


class LstmMultiAtt(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, attention_size, attention_hops, drop_out):
        super(LstmMultiAtt, self).__init__()
        self.lstm_layer = LstmLayer(input_size, lstm_hidden_size, num_layers=lstm_num_layers)

        self.att_layer = MultiAttentionLayer(2 * lstm_hidden_size, attention_size,
                                             attention_hops=attention_hops)
        self.drop_out = drop_out
        if drop_out > 0:
            self.drop_out_layer = nn.Dropout(drop_out)

    def forward(self, sequence_input, seq_lengths):
        """

        :param sequence_input: embedded products, [batch_size, max_word, emb_size]
        :param seq_lengths:
        :return:
        """
        # seq_lengths_clone = seq_lengths.clone()
        # seq_lengths_clone[seq_lengths == 0] = 1

        lstm_output, _ = self.lstm_layer(sequence_input, seq_lengths)
        if torch.isnan(lstm_output).sum() > 0:
            print('sequence_input', sequence_input)
            print('seq_lengths', seq_lengths)
            print('lstm_output', lstm_output)
            raise ValueError('lstm_output have Nan')
        # create mask
        # batch_size, max_len = sequence_input.shape[:-1]
        # mask = torch.arange(max_len).expand(batch_size, max_len).cuda() < seq_lengths.unsqueeze(1)
        # mask = mask.unsqueeze(-1)
        #
        # lstm_output = lstm_output * mask.float()

        att_output, attention_scores = self.att_layer(lstm_output, seq_lengths)

        # att_output[seq_lengths == 0] = 0
        if torch.isnan(att_output).sum() > 0:
            print('sequence_input', sequence_input)
            print('seq_lengths', seq_lengths)
            print('att_output', att_output)
            for a in att_output:
                print(a)
            raise ValueError('att_output have Nan')

        if self.drop_out > 0:
            att_output = self.drop_out_layer(att_output)

        return att_output, attention_scores


def filter_sents(flat_document, flat_sequence_lengths):
    return flat_document[flat_sequence_lengths > 0], flat_sequence_lengths[flat_sequence_lengths > 0]


def make_tokens_att_full(vectors, lengths, batch_size, max_sent):
    """Padding to all document have same len."""
    # print("vectors.shape", vectors.shape)
    if torch.cuda.is_available():
        out_tokens_att = torch.zeros(batch_size, max_sent, vectors.shape[-1]).cuda()
    else:
        out_tokens_att = torch.zeros(batch_size, max_sent, vectors.shape[-1])
    # print("out_tokens.shape", out_tokens_att.shape)
    # print(lengths.tolist())

    pre_num_sent = 0
    for idx, num_sent in enumerate(lengths):
        # print("-----------")
        # print("idx, num_sent, max_sent, pre_num_sent", idx, num_sent.tolist(), max_sent, pre_num_sent)
        # print(idx*max_sent, idx*max_sent+num_sent)
        out_tokens_att[idx, :num_sent] = vectors[pre_num_sent:pre_num_sent + num_sent]
        pre_num_sent += num_sent
    # print('out_tokens_att.requires_grad', out_tokens_att.requires_grad)
    return out_tokens_att


class BaseModel(nn.Module):
    def __init__(self, opt, num_embedding_chars):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.char_embeddings = CharacterEmbedding(opt['char_embedding_dim'], num_embedding_chars)

        self.word_level = LstmMultiAtt(opt['char_embedding_dim'], opt['lstm_hidden_size'], opt['lstm_num_layers'],
                                       opt['attention_size'],
                                       # attention_hops=1,
                                       attention_hops=opt['attention_hops'],
                                       drop_out=opt['drop_out'])

        self.sen_level = LstmMultiAtt(2 * opt['lstm_hidden_size'], opt['lstm_hidden_size'], opt['lstm_num_layers'],
                                      opt['attention_size'], attention_hops=opt['attention_hops'],
                                      drop_out=opt['drop_out'])

    def forward(self, products, sen_lens, word_lens):
        batch_size, max_word, max_char = products.shape
        flatten_product = products.flatten(end_dim=-2)
        flatten_word_lens: torch.Tensor = word_lens.flatten()

        flatten_product, flatten_word_lens = filter_sents(flatten_product, flatten_word_lens)

        flatten_products_emb: torch.Tensor = self.char_embeddings(flatten_product)
        word_vectors, word_attention_scores = self.word_level(flatten_products_emb, flatten_word_lens)
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

        return sen_vectors, (word_attention_scores, sen_attention_scores)


class CharacterWordModel(nn.Module):
    def __init__(self, opt, num_embedding_chars):
        super(CharacterWordModel, self).__init__()
        self.basemodel = BaseModel(opt, num_embedding_chars)

    def forward(self, products, sen_lens, word_lens):
        batch_size = int(products.shape[0] / 2)

        sentence_vectors, _ = self.basemodel(products, sen_lens, word_lens)
        sentences_supplier, sentences_core_product = torch.split(sentence_vectors, batch_size, dim=0)

        return sentences_supplier, sentences_core_product


class ManhattanCharWordModel(nn.Module):
    def __init__(self, opt, num_embedding_chars):
        super(ManhattanCharWordModel, self).__init__()
        self.basemodel = BaseModel(opt, num_embedding_chars)

    def forward(self, products, sen_lens, word_lens, is_predict=False):

        sentence_vectors, _ = self.basemodel(products, sen_lens, word_lens)
        if not is_predict:
            batch_size = int(products.shape[0] / 2)
            sentences_supplier, sentences_core_product = torch.split(sentence_vectors, batch_size, dim=0)
            # print(torch.norm(sentences_supplier - sentences_core_product, 1, dim=-1))
            # print(sentences_supplier, sentences_core_product)
        else:
            batch_size = products.shape[0] - 1
            sentences_supplier: torch.Tensor = sentence_vectors[:1]
            sentences_supplier.repeat((batch_size, 1))
            sentences_core_product = sentence_vectors[1:]

        return torch.exp(-torch.norm(sentences_supplier - sentences_core_product, 1, dim=-1))
