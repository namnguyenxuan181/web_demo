import torch
from torch import nn
from . import LstmLayer, MultiAttentionLayer, CharacterEmbedding


class BaseModel(nn.Module):
    def __init__(self, opt, num_embedding_chars):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.char_embeddings = CharacterEmbedding(opt['char_embedding_dim'], num_embedding_chars)
        self.lstm_layer = LstmLayer(opt['char_embedding_dim'], opt['lstm_hidden_size'],
                                    num_layers=opt['lstm_num_layers'])
        self.att_layer = MultiAttentionLayer(2 * opt['lstm_hidden_size'], opt['attention_size'],
                                             attention_hops=opt['attention_hops'])

    def forward(self, products: torch.Tensor, lengths: torch.Tensor):
        batch_size = lengths.shape[0]
        assert batch_size % 2 == 0
        products_embedding = self.char_embeddings(products)
        lstm_outputs, _ = self.lstm_layer(products_embedding, lengths)
        sentence_vector, attention_scores = self.att_layer(lstm_outputs, lengths)

        sentences_supplier, sentences_core_product = torch.split(sentence_vector, int(batch_size / 2), dim=0)

        return sentences_supplier, sentences_core_product


class CharacterModel(nn.Module):
    """
    This class concate s_A and s_B then push it to fc layers. Then soft_max.
    """

    def __init__(self, opt, num_embedding, output_size):
        super(CharacterModel, self).__init__()
        self.opt = opt
        self.base_model = BaseModel(opt, num_embedding)

        if opt['drop_out'] > 0:
            self.drop_out_layer = nn.Dropout(opt['drop_out'])
        self.fc_layer = nn.Linear(4 * opt['lstm_hidden_size'], opt['fc_size'])
        self.final_linear = nn.Linear(opt['fc_size'], output_size)

    def classifier(self, x):
        if self.opt['drop_out'] > 0:
            x = self.drop_out_layer(x)

        x = self.fc_layer(x)
        x = self.final_linear(x)
        return x

    def forward(self, products: torch.Tensor, lengths: torch.Tensor):

        sentences_supplier, sentences_core_product = self.base_model(products, lengths)

        pair_vector = torch.cat([sentences_supplier, sentences_core_product], dim=-1)

        logits = self.classifier(pair_vector)
        return logits
