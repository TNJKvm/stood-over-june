import argparse
import os.path
import sys

sys.path.append('./')

import torch
from tqdm import tqdm

import src.models
import src.datasets.datasets
from src.utils.selection import uniform_selection
from src.utils.trainer import Trainer
from src.utils.config import load_config
from src.utils.reproducibillity import set_seed

CONFIG = None


def main():
    pbar = tqdm(total=len(CONFIG['seeds']) * CONFIG['number_datasets'], colour='green')
    for seed in CONFIG['seeds']:
        set_seed(seed)
        for i in range(CONFIG['number_datasets']):
            data \
                = getattr(src.datasets.datasets, CONFIG["dataset"][i])(CONFIG["data_path"][i], CONFIG["batch_size"][i])

            model = getattr(src.models, CONFIG["model"][i])(classes=100) \
                if i == 0 else getattr(src.models, CONFIG["model"][i])()

            trainer = Trainer(model, data[CONFIG["train_set"]], data[2], selection=uniform_selection, train_percent=1,
                              device=DEVICE)

            best_model, best_epoch, best_loss, best_accuracy, _results \
                = trainer.train_until_fitted(check_last=CONFIG['check_last'], max_epochs=CONFIG['max_epochs'])
            tqdm.write(f"Best model after {best_epoch} epochs with accuracy {best_accuracy} and loss {best_loss}")

            filename = f"irr_{CONFIG['dataset_name'][i]}_{CONFIG['model_name']}_{seed}{CONFIG.get('file_suffix'), ''}.pt"
            torch.save(best_model.state_dict(), os.path.join(CONFIG['out_path'], filename))

            if not os.path.exists(CONFIG['result_csv']):
                with open(CONFIG['result_csv'], "w") as result_file:
                    result_file.write(f"filename;epoch;accuracy;loss")

            with open(CONFIG['result_csv'], "a") as result_file:
                result_file.write(f"\n{filename};{best_epoch};{best_accuracy};{best_loss}")
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
                           metavar='gpu')

    args = my_parser.parse_args()

    return vars(args)


if __name__ == '__main__':
    arguments = parse_arguments()
    CONFIG = load_config(arguments['path_config'])
    DEVICE = None if arguments['gpu'] is None else torch.device('cuda', arguments['gpu'])
    main()
