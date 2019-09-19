import os
import json
import pickle


def save_args(args, save_dir):
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as fo:
        json.dump(vars(args), fo, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, 'args.pickle'), 'wb') as fo:
        pickle.dump(args, fo)


def load_args(save_dir):
    with open(os.path.join(save_dir, 'config.json'), 'r', encoding='utf-8') as fi:
        print(f"config:\n{fi.read()}")

    with open(os.path.join(save_dir, 'args.pickle'), 'rb') as fi:
        return pickle.load(fi)


def get_best_model(folder_path, index, reverse=True):
    def key_func(fpath):
        fpath = fpath.replace('.pth', '')
        metric = fpath.split('_')[index]
        print(metric)
        metric = metric.split(':')[1]
        return float(metric)
    models = list(p for p in os.listdir(folder_path) if p.endswith('.pth'))
    sorted_models = sorted(models, key=key_func, reverse=reverse)
    # sorted_models = sorted(saved_models, key=key_func, reverse=False)
    print("top 2 saved_models", sorted_models[:2])

    return os.path.join(folder_path, sorted_models[0])
