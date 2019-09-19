from ..models.cnn_model import TripletCNN
from .charword_ranking import TripletCharWordTrainer


class CNNTrainer(TripletCharWordTrainer):
    def prepare_model(self, opt, is_train=True):
        from ..char_utils import CHAR_LIST_KIND_1

        num_embedding = len(CHAR_LIST_KIND_1)

        self.model = TripletCNN(opt=vars(opt), num_embedding_chars=num_embedding, p=self.loss_p)
