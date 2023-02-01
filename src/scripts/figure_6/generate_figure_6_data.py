import pickle
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd())))

from src.datasets import datasets
from src.models.simple.MLP import MLP512
from src.utils.selection import irreducible_loss_selection, reducible_loss_selection, uniform_selection, loss_selection, \
    grad_norm_selection
from src.utils.trainer import Trainer
from tqdm import tqdm

EPOCHS = 100
REPEATS = 5
BATCH_SIZE = 160
DATA_PATH = ''
EXPERIMENT_SELECTIONS = [('Reducible Loss', reducible_loss_selection),
                         ('Irreducible Loss', irreducible_loss_selection),
                         ('Loss', loss_selection),
                         ('Uniform Sampling', uniform_selection),
                         ('Gradient Norm', grad_norm_selection)]


def main():
    # 1. Get the Data
    train_loader, val_loader, test_loader = datasets.get_QMNIST(DATA_PATH, BATCH_SIZE)

    # 2. Train the IRR Model
    irr_model = MLP512()
    irr_trainer = Trainer(irr_model, train_loader, test_loader)

    train_acc_list = []
    for epoch in tqdm(range(EPOCHS), position=0, leave=True, colour='red'):
        train_loss, train_acc = irr_trainer.train_log()
        train_acc_list += train_acc
        tqdm.write(f"Train Epoch     {epoch}     Loss: {train_loss:.2f}")
        val_loss, val_accuracy, correct, length = irr_trainer.validate()
        tqdm.write(
            f"Validate Epoch  {epoch}     Loss: {val_loss:.2f}    Accuracy: {val_accuracy:.2f}    Correct: {correct}/{length}")

    trained_irr_model = irr_trainer.get_model_GPU()

    steps_complete = REPEATS * EPOCHS * len(EXPERIMENT_SELECTIONS) * 4
    pbar = tqdm(desc='Total progress:', total=steps_complete, colour='green')

    # 3. Train the target model
    # Run experiment for No noise
    experiment_result = {
        "selection_method": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    for experiment, selection_method in EXPERIMENT_SELECTIONS:
        for iteration in range(REPEATS):
            tqdm.write(f"Applying {experiment} Selection {iteration}/{REPEATS} with no noise: ")
            target_model = MLP512()
            target_trainer = Trainer(target_model, val_loader, test_loader, selection=selection_method,
                                     irr_model=trained_irr_model)

            target_train_acc_list = []
            target_val_acc_list = []

            for epoch in range(EPOCHS):
                train_loss, train_acc, val_acc = target_trainer.train_step_validate()
                target_train_acc_list += train_acc
                target_val_acc_list += val_acc
                tqdm.write(f"Train Target Epoch     {epoch}     Loss: {train_loss:.2f}")
                val_loss, val_accuracy, correct, length = target_trainer.validate()
                tqdm.write(
                    f"Validate Target Epoch  {epoch}     Loss: {val_loss:.2f}    Accuracy: {val_accuracy:.2f}    Correct: {correct}/{length}")
                pbar.update(1)

            experiment_result["selection_method"].append(experiment)
            experiment_result["train_accuracy"].append(target_train_acc_list)
            experiment_result["val_accuracy"].append(target_val_acc_list)

    with open("figure_1_no_noise.pickle", "wb") as file:
        pickle.dump(experiment_result, file)

    # Run experiment for unstructured label noise
    experiment_result = {
        "selection_method": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    for experiment, selection_method in EXPERIMENT_SELECTIONS:
        for iteration in range(REPEATS):
            tqdm.write(f"Applying {experiment} Selection {iteration}/{REPEATS} with struct label noise: ")

            noisy_train_data = datasets.get_MNIST_label_noise(DATA_PATH, BATCH_SIZE, 0.1)

            target_model = MLP512()
            target_trainer = Trainer(target_model, noisy_train_data, test_loader, selection=selection_method,
                                     irr_model=trained_irr_model)

            target_train_acc_list = []
            target_val_acc_list = []

            for epoch in range(EPOCHS):
                train_loss, train_acc, val_acc = target_trainer.train_step_validate()
                target_train_acc_list += train_acc
                target_val_acc_list += val_acc
                tqdm.write(f"Train Target Epoch     {epoch}     Loss: {train_loss:.2f}")
                val_loss, val_accuracy, correct, length = target_trainer.validate()
                tqdm.write(
                    f"Validate Target Epoch  {epoch}     Loss: {val_loss:.2f}    Accuracy: {val_accuracy:.2f}    Correct: {correct}/{length}")
                pbar.update(1)

            experiment_result["selection_method"].append(experiment)
            experiment_result["train_accuracy"].append(target_train_acc_list)
            experiment_result["val_accuracy"].append(target_val_acc_list)

    with open("figure_1_unstruct_noise.pickle", "wb") as file:
        pickle.dump(experiment_result, file)

    # Run experiment for struct label noise
    experiment_result = {
        "selection_method": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    for experiment, selection_method in EXPERIMENT_SELECTIONS:
        for iteration in range(REPEATS):
            tqdm.write(f"Applying {experiment} Selection {iteration}/{REPEATS} with struct label noise: ")

            noisy_train_data = datasets.get_MNIST_struct_noise(DATA_PATH, BATCH_SIZE, 0.5)

            target_model = MLP512()
            target_trainer = Trainer(target_model, noisy_train_data, test_loader, selection=selection_method,
                                     irr_model=trained_irr_model, pbar=pbar)

            target_train_acc_list = []
            target_val_acc_list = []

            for epoch in range(EPOCHS):
                train_loss, train_acc, val_acc = target_trainer.train_step_validate()
                target_train_acc_list += train_acc
                target_val_acc_list += val_acc
                tqdm.write(f"Train Target Epoch     {epoch}     Loss: {train_loss:.2f}")
                val_loss, val_accuracy, correct, length = target_trainer.validate()
                tqdm.write(
                    f"Validate Target Epoch  {epoch}     Loss: {val_loss:.2f}    Accuracy: {val_accuracy:.2f}    Correct: {correct}/{length}")
                pbar.update(1)

            experiment_result["selection_method"].append(experiment)
            experiment_result["train_accuracy"].append(target_train_acc_list)
            experiment_result["val_accuracy"].append(target_val_acc_list)

    with open("figure_1_struct_noise.pickle", "wb") as file:
        pickle.dump(experiment_result, file)

    # Run experiment for ambigous mnist noise
    experiment_result = {
        "selection_method": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    ambiguous_train = datasets.get_MNIST_ambiguous('/home/alsch/Datasets/Ambiguous-MNIST', BATCH_SIZE)

    for experiment, selection_method in EXPERIMENT_SELECTIONS:
        for iteration in range(REPEATS):
            tqdm.write(f"Applying {experiment} Selection {iteration}/{REPEATS} with ambiguous noise: ")

            target_model = MLP512()
            target_trainer = Trainer(target_model, ambiguous_train, test_loader, selection=selection_method,
                                     irr_model=trained_irr_model)

            target_train_acc_list = []
            target_val_acc_list = []

            for epoch in range(EPOCHS):
                train_loss, train_acc, val_acc = target_trainer.train_step_validate()
                target_train_acc_list += train_acc
                target_val_acc_list += val_acc
                tqdm.write(f"Train Target Epoch     {epoch}     Loss: {train_loss:.2f}")
                val_loss, val_accuracy, correct, length = target_trainer.validate()
                tqdm.write(
                    f"Validate Target Epoch  {epoch}     Loss: {val_loss:.2f}    Accuracy: {val_accuracy:.2f}    Correct: {correct}/{length}")
                pbar.update(1)

            experiment_result["selection_method"].append(experiment)
            experiment_result["train_accuracy"].append(target_train_acc_list)
            experiment_result["val_accuracy"].append(target_val_acc_list)

    with open("figure_1_ambiguous_noise.pickle", "wb") as file:
        pickle.dump(experiment_result, file)


if __name__ == "__main__":
    print(sys.argv)
    BATCH_SIZE, REPEATS, DATA_PATH, EPOCHS = sys.argv[1:]
    BATCH_SIZE, REPEATS, EPOCHS = int(BATCH_SIZE), int(REPEATS), int(EPOCHS)
    main()
