import argparse
import os.path
import pickle
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd())))
sys.path.append('./')

import torch
from tqdm import tqdm

from src.datasets import datasets
import src.models
import src.datasets.datasets
from src.utils.selection import reducible_loss_selection, uniform_selection
from src.utils.trainer import Trainer
from src.utils.config import load_config
from src.utils.reproducibillity import set_seed, load_model

CONFIG = None
ARGUMENTS = None


def main():
    device = torch.device('cuda', ARGUMENTS['gpu'])
    torch.cuda.device(device)
    baseline = True

    if not ARGUMENTS['train_irr'] and ARGUMENTS.get('irr_model') in os.listdir(CONFIG['irr_model']):
        irr_model, model_type, seed, dataset = load_model(ARGUMENTS['irr_model'])
        baseline = False

    set_seed(CONFIG['seed'])
    dataset_train, dataset_val, dataset_test = datasets.get_Cloting1M(CONFIG["data_path"], CONFIG["batch_size"])
    model = getattr(src.models, f'{ARGUMENTS["model"]}_Clothing')()
    print(f"Memory usage: {torch.cuda.memory_allocated()}")

    if ARGUMENTS['train_irr']:
        trainer = Trainer(model, dataset_train, dataset_test, selection=uniform_selection, train_percent=1.0,
                          device=device)
        filename = f"irr_Clothing1M_Resnet18_{CONFIG['seed']}.pt"
        tqdm.write('Mode: IRR Train')
    elif baseline:
        trainer = Trainer(model, dataset_val, dataset_test, selection=uniform_selection, train_percent=0.1,
                          device=device)
        filename = f"baseline_Clothing1M_{ARGUMENTS['model']}_{CONFIG['seed']}.pt"
        tqdm.write('Mode: Baseline Train')
    else:
        trainer = Trainer(model, dataset_val, dataset_test, selection=reducible_loss_selection, train_percent=0.1,
                          device=device, irr_model=irr_model)
        filename = f"target_Clothing1M_{ARGUMENTS['model']}_{CONFIG['batch_size']}_{CONFIG['seed']}.pt"
        tqdm.write('Mode: Target Train')

    print(f"Memory usage: {torch.cuda.memory_allocated() / 1e6}")

    best_model, best_epoch, best_loss, best_accuracy, results \
        = trainer.train_until_fitted(check_last=CONFIG['check_last'],
                                     max_epochs=CONFIG['max_epochs'],
                                     min_epochs=CONFIG['min_epochs'])

    tqdm.write(f"Best model after {best_epoch} epochs with accuracy {best_accuracy} and loss {best_loss}")

    torch.save(best_model.state_dict(), os.path.join(CONFIG['out_path'], filename))

    csv_result = os.path.join(CONFIG['out_path'], "results_clothing-1M.csv")

    with open(os.path.join(CONFIG['out_path'], filename.split('.')[0] + '.pickle'), 'wb') as file:
        pickle.dump(results, file)

    if not os.path.exists(csv_result):
        with open(csv_result, "w") as result_file:
            result_file.write(f"filename;epoch;accuracy;loss")
    with open(csv_result, "a") as result_file:
        result_file.write(f"\n{filename};{best_epoch};{best_accuracy};{best_loss}")


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
                           default=0,
                           required=False,
                           metavar='gpu')

    my_parser.add_argument('-m',
                           '--model',
                           action='store',
                           type=str,
                           required=True,
                           metavar='model')

    my_parser.add_argument('-i',
                           '--irr_model',
                           action='store',
                           type=str,
                           required=False,
                           metavar='irr_model')

    my_parser.add_argument('-t',
                           '--train_irr',
                           action='store',
                           type=bool,
                           default=False,
                           required=False,
                           metavar='train_irr')

    args = my_parser.parse_args()

    return vars(args)


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    CONFIG = load_config(ARGUMENTS['path_config'])
    main()
