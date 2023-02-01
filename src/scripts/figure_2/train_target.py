import argparse
import glob
import os.path
import pickle
import sys

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
DEVICE = None

def main():
    irr_models = glob.glob(os.path.join(CONFIG["irr_model_path"], "*.pt"))
    pbar = tqdm(total=len(irr_models))
    for irr_model in irr_models:

        irr_model, model_type, seed, dataset = load_model(irr_model)
        set_seed(seed)
        data = getattr(src.datasets.datasets, f'get_{dataset}')(os.path.join(CONFIG["data_path"], dataset),
                                                                CONFIG["batch_size"])

        model = src.models.ResNet18_Imagenet(classes=100) \
            if dataset == 'CIFAR100' else src.models.ResNet18()

        trainer = Trainer(model, data[1], data[2], selection=reducible_loss_selection, train_percent=0.1,
                          irr_model=irr_model, device=DEVICE)

        best_model, best_epoch, best_loss, best_accuracy, results \
            = trainer.train_until_fitted(check_last=CONFIG['check_last'],
                                         max_epochs=CONFIG['max_epochs'],
                                         check_add=CONFIG['add_epochs'])
        tqdm.write(f"Best model after {best_epoch} epochs with accuracy {best_accuracy} and loss {best_loss}")

        filename = f"target_{dataset}_Resnet18_{seed}.pt"
        torch.save(best_model.state_dict(), os.path.join(CONFIG['out_path'], filename))

        if not os.path.exists(CONFIG['result']):
            with open(CONFIG['result'], "w") as result_file:
                result_file.write(f"filename;epoch;accuracy;loss")
        with open(CONFIG['result'], "a") as result_file:
            result_file.write(f"\n{filename};{best_epoch};{best_accuracy};{best_loss}")

        with open(os.path.join(CONFIG['out_path'], f"target_{dataset}_Resnet18_{seed}.pickle"), 'wb') as file:
            pickle.dump(results, file)
        pbar.update(1)


def parse_arguments():
    my_parser = argparse.ArgumentParser(description='Starts a baseline experiment')

    my_parser.add_argument('-p',
                           '--path_config',
                           action='store',
                           type=str,
                           required=True,
                           metavar='path_config')

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
    arguments = parse_arguments()
    CONFIG = load_config(arguments['path_config'])
    DEVICE = None if arguments['gpu'] is None else torch.device('cuda', arguments['gpu'])
    main()
