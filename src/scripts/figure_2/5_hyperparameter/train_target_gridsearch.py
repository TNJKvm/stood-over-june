import argparse
import glob
import itertools
import os.path
import pickle
import sys
from pathlib import Path

sys.path.append('./')

import torch
from tqdm import tqdm

import src.models
import src.datasets.datasets
from src.utils.selection import reducible_loss_selection
from src.utils.trainer import Trainer
from src.utils.config import load_config
from src.utils.reproducibillity import set_seed, load_model

CONFIG = None
ARGUMENTS = None
DEVICE = None
batch_sizes = [160, 320, 960]
learning_rate = [0.0001, 0.001, 0.01]
weight_decay = [0.001, 0.01, 0.1]

PARAMS = list(itertools.product(*[batch_sizes, learning_rate, weight_decay]))


def main():
    params = PARAMS[ARGUMENTS['from']:ARGUMENTS['to']]
    pbar = tqdm(total=len(params))
    for param in params:
        irr_model, model_type, seed, dataset = load_model(CONFIG["irr_model_path"])
        set_seed(seed)

        data = getattr(src.datasets.datasets, f'get_{dataset}')(os.path.join(CONFIG["data_path"], dataset), param[0])

        model = src.models.ResNet18_Imagenet(classes=100) \
            if dataset == 'CIFAR100' else src.models.ResNet18()

        trainer = Trainer(model, data[1], data[2],
                          selection=reducible_loss_selection,
                          train_percent=0.1,
                          irr_model=irr_model,
                          device=DEVICE,
                          learning_rate=param[1],
                          weight_decay=param[2])

        best_model, best_epoch, best_loss, best_accuracy, results \
            = trainer.train_until_fitted(check_last=CONFIG['check_last'],
                                         max_epochs=CONFIG['max_epochs'],
                                         check_add=CONFIG['add_epochs'])

        tqdm.write(f"Best model after {best_epoch} epochs with accuracy {best_accuracy} and loss {best_loss}")

        filename = f"target_{dataset}_Resnet18_{seed}_{param[0]}_{str(param[1]).split('.')[1]}_{str(param[2]).split('.')[1]}.pt"

        torch.save(best_model.state_dict(), os.path.join(CONFIG['out_path'], filename))

        if not os.path.exists(os.path.join(CONFIG['out_path'], CONFIG['result_csv'])):
            with open(os.path.join(CONFIG['out_path'], CONFIG['result_csv']), "w") as result_file:
                result_file.write(f"filename;epoch;accuracy;loss")
        with open(os.path.join(CONFIG['out_path'], CONFIG['result_csv']), "a") as result_file:
            result_file.write(f"\n{filename};{best_epoch};{best_accuracy};{best_loss}")

        with open(os.path.join(CONFIG['out_path'], f"target_{dataset}_Resnet18_{seed}_{param[0]}_{str(param[1]).split('.')[1]}_{str(param[2]).split('.')[1]}.pickle"), 'wb') as file:
            pickle.dump(results, file)
        pbar.update(1)


def parse_arguments():
    my_parser = argparse.ArgumentParser(description='Starts a gridsearch experiment')

    my_parser.add_argument('-p',
                           '--path_config',
                           action='store',
                           type=str,
                           required=True,
                           metavar='path_config')

    my_parser.add_argument('-f',
                           '--from',
                           action='store',
                           type=int,
                           required=False,
                           default=0,
                           metavar='from')

    my_parser.add_argument('-t',
                           '--to',
                           action='store',
                           type=int,
                           required=False,
                           default=0,
                           metavar='from')

    my_parser.add_argument('-g',
                           '--gpu',
                           action='store',
                           type=int,
                           required=False,
                           default=None,
                           metavar='gpu')

    args = my_parser.parse_args()

    return vars(args)


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    CONFIG = load_config(ARGUMENTS['path_config'])
    DEVICE = None if ARGUMENTS['gpu'] is None else torch.device('cuda', ARGUMENTS['gpu'])
    main()
