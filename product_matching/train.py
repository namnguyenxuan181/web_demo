import torch
import os
import numpy as np
import random

CURRENT_DIR = os.path.dirname(__file__)
BASENAME_DIR = os.path.basename(CURRENT_DIR)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def main(opt):
    set_seed(opt.seed)

    if opt.model == 'char':
        from product_matching.trainers.char_trainer import CharacterTrainer as Trainer
    elif opt.model == 'charword':
        from product_matching.trainers.char_word_trainer import CharWordTrainer as Trainer
    elif opt.model == 'manhattan-charword':
        from product_matching.trainers.char_word_trainer import ManhattanCharWordTrainer as Trainer
    elif opt.model == 'triplet-charword':
        from product_matching.trainers.charword_ranking import TripletCharWordTrainer as Trainer
    elif opt.model == 'triplet-charword2':
        from product_matching.trainers.charword_ranking import TripletCharWord2Trainer as Trainer
    elif opt.model == 'triplet-charword3':
        from product_matching.trainers.charword_ranking import TripletCharWord3Trainer as Trainer
    elif opt.model == 'triplet-cnn':
        from product_matching.trainers.cnn_trainer import CNNTrainer as Trainer
    else:
        raise ValueError

    if not opt.run_test:
        trainer = Trainer(opt, kind='train')
        trainer.train()
        save_dir = trainer.save_dir
        del trainer
    else:
        save_dir = opt.save_dir
    trainer = Trainer.load_from_saved_dir(save_dir, kind='eval')
    trainer.evaluate()
    trainer.analysis(save_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['char', 'charword', 'manhattan-charword',
                                                      'triplet-charword', 'triplet-charword2',
                                                      'triplet-charword3', 'triplet-cnn'],
                        default='char')
    parser.add_argument('--save_dir', type=str, )

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--count_backward', type=int, default=4)
    parser.add_argument("--epoch", type=int, default=25)

    parser.add_argument('--data_kind', type=int, choices=[0, 1], default=0)
    parser.add_argument("--char_dim", dest='char_embedding_dim', type=int, default=200, help='Embedding size')
    parser.add_argument('--lstm_h_size', dest='lstm_hidden_size', type=int, default=256, help='LSTM size')
    parser.add_argument('--lstm_n_layers', dest='lstm_num_layers', type=int, default=1, help='Number of lstm layers')
    parser.add_argument("--att_size", dest='attention_size', type=int, default=64, help='Attention size')
    parser.add_argument('--att_hops', dest='attention_hops', type=int, default=1, help='Number attention hops')
    parser.add_argument('--fc_size', type=int, default=128, help='Full connected size.')
    parser.add_argument('--drop_out', default=0.0, type=float, help='Drop out for last fc')
    # parser.add_argument('--drop_out', type=float, default=0.0, nargs='+', help='Drop out for last fc')

    parser.add_argument('--triplet_norm_p', type=float, default=2.0, help='Norm kind for triplet, p=1 or 2.')
    parser.add_argument('--triplet_margin', type=float, default=1.0, help='Margin for triplet.')
    parser.add_argument('--norm_e', action='store_true', help='Normalize vector presenting a sentence.')

    parser.add_argument('--penalty_ratio', type=float, nargs='+', help='penalty_ratio for custom loss')

    # CNN
    parser.add_argument('--kernel_sizes', type=int, nargs='+', help='list kernel_size', default=[2, 3, 4, 5])
    parser.add_argument('--kernel_num', type=int, help='Number kernel', default=100)

    parser.add_argument('--optim', type=str, choices=['adam', 'rmsprop'], default='rmsprop')

    parser.add_argument('--run_test', action='store_true', )

    opt = parser.parse_args()
    main(opt)
