import json
import os
import pickle
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd())))

import torch
from tqdm import tqdm

from src.datasets import datasets
from src.models.Resnet import ResNet18_Imagenet
from src.utils.reproducibillity import set_seed
from src.utils.selection import uniform_selection
from src.utils.trainer import Trainer


# Generate baseline for CIFAR-10 using Resnet18

def main(data_path, output_file, train_percent=0.1, batch_size=320, epochs=150, gpu=0, seeds=[42]):
    device = torch.device(f'cuda:{gpu}')

    results = {
        'seed': [],
        'result': []
    }

    for idx, seed in enumerate(seeds):

        set_seed(seed)

        train_loader, val_loader, test_loader = datasets.get_CIFAR100(data_path, batch_size)

        model = ResNet18_Imagenet(classes=100)
        trainer = Trainer(model, val_loader, test_loader, selection=uniform_selection, train_percent=train_percent, device=device)

        model_result = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        prefix = f"Model {idx+1}/{len(seeds)}"

        for epoch in tqdm(range(epochs), position=0, leave=True, colour='blue', desc=prefix):
            train_loss, train_acc = trainer.train_log()
            model_result['train_acc'].append(train_acc.pop())
            model_result['train_loss'].append(train_loss)
            tqdm.write(f"Train Epoch     {epoch}     Loss: {train_loss:.2f}")
            val_loss, val_accuracy, correct, length = trainer.validate()
            tqdm.write(
                f"Validate Epoch  {epoch}     Loss: {val_loss:.2f}    Accuracy: {val_accuracy:.2f}    Correct: {correct}/{length}")
            model_result['val_acc'].append(val_accuracy.data.item())
            model_result['val_loss'].append(val_loss)

        results['seed'].append(seed)
        results['result'].append(model_result)
        filename = f"baseline_CIFAR100_Resnet18_{seed}.pt"
        torch.save(trainer.model.state_dict(), filename)

    with open(output_file, 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    config_path = sys.argv[1]
    gpu = int(sys.argv[2])

    with open(config_path) as f:
        CONFIG = json.load(f)

    main(data_path=CONFIG['data_path'], output_file=os.path.join(CONFIG['out_path'], 'cifar10_baseline.pickle'),
         batch_size=CONFIG['batch_size'],
         epochs=CONFIG['epochs'], gpu=gpu, seeds=CONFIG['seeds'])
