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
from src.utils.selection import reducible_loss_selection, uniform_selection
from src.utils.trainer import Trainer
from src.utils.config import load_config
from src.utils.reproducibillity import set_seed, load_model

CONFIG = None
ARGUMENTS = None


def main():
    anmount_trainings = len(CONFIG['seeds']) * len(CONFIG['datasets'])

    pbar = tqdm(total=anmount_trainings)

    for seed in CONFIG['seeds']:
        set_seed(seed)
        for dataset in CONFIG['datasets']:
            data = getattr(src.datasets.datasets, f'get_{dataset}')(os.path.join(CONFIG["data_path"], dataset),
                                                                    CONFIG["batch_size"])

            # Train target model
            model_path = os.path.join(CONFIG['irr_model_path'], f"irr_{dataset}_CNN_{seed}.pt")
            irr_model, model_type, seed, dataset = load_model(model_path)

            model = get_model_by_reference(ARGUMENTS['model'], dataset)

            trainer = Trainer(model, data[1], data[2], selection=reducible_loss_selection, train_percent=0.1,
                              irr_model=irr_model, device=torch.device('cuda', ARGUMENTS['gpu']))

            best_model, best_epoch, best_loss, best_accuracy, results \
                = trainer.train_until_fitted(check_last=CONFIG['check_last'],
                                             max_epochs=CONFIG['max_epochs'],
                                             check_add=CONFIG['add_epochs'])
            tqdm.write(f"Best model after {best_epoch} epochs with accuracy {best_accuracy} and loss {best_loss}")

            filename = f"target_{dataset}_{ARGUMENTS['model']}_{seed}.pt"
            torch.save(best_model.state_dict(), os.path.join(CONFIG['out_path'], filename))

            results_csv_path = os.path.join(CONFIG['out_path'], f"target_{ARGUMENTS['model']}_training.csv")

            if not os.path.exists(results_csv_path):
                with open(results_csv_path, "w") as result_file:
                    result_file.write(f"filename;epoch;accuracy;loss")
            with open(results_csv_path, "a") as result_file:
                result_file.write(f"\n{filename};{best_epoch};{best_accuracy};{best_loss}")

            with open(os.path.join(CONFIG['out_path'], f"target_{dataset}_{ARGUMENTS['model']}_{seed}.pickle"),
                      'wb') as file:
                pickle.dump(results, file)
            pbar.update(1)


def get_model_by_reference(model_ref: str, dataset_ref: str):
    args = {}
    if model_ref.startswith('ResNet'):
        model_ref = model_ref + '_Imagenet'
    if dataset_ref == 'CIFAR100':
        args['num_classes'] = 100
    model = getattr(src.models, model_ref)(**args)
    return model


def parse_arguments():
    my_parser = argparse.ArgumentParser(description='Starts a baseline experiment')

    my_parser.add_argument('-p',
                           '--path_config',
                           action='store',
                           type=str,
                           required=True,
                           metavar='path_config')

    my_parser.add_argument('-m',
                           '--model',
                           choices=['ResNet18', 'ResNet34', 'ResNet50', 'densenet121',
                                    'googlenet', 'inception_v3', 'mobilenet_v2', 'vgg11'],
                           action='store',
                           type=str,
                           required=True,
                           metavar='model')

    my_parser.add_argument('-g',
                           '--gpu',
                           action='store',
                           type=int,
                           default=0,
                           required=False,
                           metavar='model')

    args = my_parser.parse_args()

    return vars(args)


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    CONFIG = load_config(ARGUMENTS['path_config'])
    #CONFIG = load_config(
    #    '/home/alsch/PycharmProjects/RHO-Loss-Reproduction/src/scripts/figure_2/1_big_model/config_target_local.json')
    main()
