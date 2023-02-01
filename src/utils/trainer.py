import copy
import gc

import torch
from torch import nn
from torch.fx.experimental.unification import variables
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from tqdm.std import tqdm as tqdm_class

from src.utils.selection import uniform_selection


class Trainer:
    def __init__(self, model: nn.Module, train_data: DataLoader, val_data: DataLoader, device: torch.device = None,
                 selection=uniform_selection, irr_model=None, irr_model_2=None, train_data_2=None, pbar=None,
                 learning_rate=0.001, train_percent=0.1, weight_decay=0.01):
        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.train_loader = train_data
        self.train_loader_2 = train_data_2
        self.val_loader = val_data
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = CrossEntropyLoss()
        self.selection = selection
        self.irr_model = None if irr_model is None else irr_model.to(self.device)
        self.irr_model_2 = None if irr_model_2 is None else irr_model_2.to(self.device)
        self.minibatch_size = int(train_data.batch_size * train_percent)
        self.pbar = pbar
        self.model_regestry = {
            'model': [],
            'accuracy': [],
            'loss': []
        }

    def train(self):
        self.model.train()

        loss = -1
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = self.apply_selction(data, target)
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        return loss.data.item()

    def apply_selction(self, data, target, irr_model=None):
        irr_model = self.irr_model if irr_model is None else irr_model
        data, target, idx_ = self.selection(data, target, self.minibatch_size, irr_model, self.model, self.device)
        return data, target

    def train_log(self):
        self.model.train()

        losses = -1
        accuracies = []
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Train: ", colour='#59c9a5', leave=False)):
            #tqdm.write(f"Batch {batch_idx} before: {round(((torch.cuda.memory_allocated() / 1e6) / 12000) * 100)}%")
            data, target = self.apply_selction(data, target)
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if str(type(output)) == "<class 'torchvision.models.googlenet.GoogLeNetOutputs'>":
                output = output.logits
            #tqdm.write(f"Batch {batch_idx} mid: {round(((torch.cuda.memory_allocated() / 1e6) / 12000) * 100)}%")
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct.to(torch.float32) / len(data)
            accuracies.append(accuracy.data.item())
            losses = loss.data.item()
            torch.cuda.empty_cache()
            #tqdm.write(f"Batch {batch_idx} after: {round(((torch.cuda.memory_allocated() / 1e6) / 12000) * 100)}%")
            #tqdm.write('')
        return losses, accuracies

    def train_step_validate(self):
        self.model.train()

        losses = -1
        accuracies = []
        val_accuracies = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = self.apply_selction(data, target)
            if isinstance(self.pbar, tqdm_class):
                self.pbar.update(1)
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct.to(torch.float32) / len(data)
            accuracies.append(accuracy.data.item())
            losses = loss.data.item()
            _val_loss, val_accuracy, _correct, _len = self.validate()
            val_accuracies.append(val_accuracy)
        return losses, accuracies, val_accuracies

    def validate(self):
        self.model.eval()
        val_loss, correct = 0, 0
        for idx, (data, target) in enumerate(tqdm(self.val_loader, desc="Validate: ", colour="#D81E5B", leave=False)):
            #tqdm.write(f"Batch {idx} validate before: {round(((torch.cuda.memory_allocated() / 1e6) / 12000) * 100)}%")
            data = data.to(self.device)
            target = target.to(self.device)
            #tqdm.write(f"Batch {idx} validate mid: {round(((torch.cuda.memory_allocated() / 1e6) / 12000) * 100)}%")
            output = self.model(data)
            val_loss += self.criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            #tqdm.write(f"Batch {idx} validate after: {round(((torch.cuda.memory_allocated() / 1e6) / 12000) * 100)}%")
            #tqdm.write('')

        val_loss /= len(self.val_loader)
        accuracy = 100. * correct.to(torch.float32) / len(self.val_loader.dataset)

        return val_loss, accuracy, correct, len(self.val_loader.dataset)

    def train_until_fitted(self,
                           min_epochs: int = 10,
                           max_epochs: int = 200,
                           validation: bool = True,
                           criterion_loss: bool = False,
                           check_last: int = 20,
                           check_add: int = 0,
                           no_holdout: bool = False,
                           early_stopping=True):

        best_model = None
        best_accuracy = 0
        best_loss = -1
        best_epoch = 0

        results_dict = {
            'accuracy': [],
            'loss': []
        }

        tripped = False

        for epoch in tqdm(range(max_epochs), desc="Overall Progress: ", colour="#fffd98"):

            if no_holdout:
                loss, accuracy = self.train_log_no_holdout()
            else:
                loss, accuracy = self.train_log()

            tqdm.write(f"Train Epoch     {epoch}     Loss: {loss:.2f}")
            if validation:
                loss, accuracy, correct, length = self.validate()
                tqdm.write(
                    f"Validate Epoch  {epoch}     Loss: {loss:.2f}    Accuracy: {accuracy:.2f}    Correct: {correct}/{length}")
                accuracy = accuracy.data.item()

            results_dict['accuracy'].append(accuracy)
            results_dict['loss'].append(loss)

            if not tripped and early_stopping:
                self.model_regestry['accuracy'].append(accuracy)
                self.model_regestry['loss'].append(loss)
                if early_stopping:
                    self.model_regestry['model'].append(self.get_model())

                if criterion_loss and best_loss > loss:
                    best_model = self.get_model()
                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch
                elif not criterion_loss and best_accuracy < accuracy:
                    best_model = self.get_model()
                    best_accuracy = accuracy
                    best_loss = loss
                    best_epoch = epoch

                if epoch > min_epochs and epoch > check_last and not early_stopping or tripped:
                    if criterion_loss:
                        if min(self.model_regestry['loss']) > best_loss:
                            if check_add == 0:
                                break
                            else:
                                tripped = True
                    else:
                        if max(self.model_regestry['accuracy']) < best_accuracy:
                            if check_add == 0:
                                break
                            else:
                                tripped = True
                if len(self.model_regestry['loss']) > check_last:
                    for key in self.model_regestry.keys():
                        self.model_regestry[key].pop(0)
            elif check_add == 0 and early_stopping:
                break
            elif early_stopping:
                check_add -= 1
        if not early_stopping:
            return self.get_model(), max_epochs, results_dict['loss'][-1], results_dict['accuracy'][-1], results_dict
        return best_model, best_epoch, best_loss, best_accuracy, results_dict

    def get_model_CPU(self):
        return copy.deepcopy(self.model).to(torch.device('cpu'))

    def get_model_GPU(self):
        return copy.deepcopy(self.model).to(self.device)

    def get_model(self):
        return copy.deepcopy(self.model)

    def train_log_no_holdout(self):
        self.model.train()

        losses = -1
        accuracies = []
        for ((data_1, target_1), (data_2, target_2)) in zip(self.train_loader, self.train_loader_2):
            data_1, target_1 = self.apply_selction(data_1, target_1)
            data_2, target_2 = self.apply_selction(data_2, target_2, self.irr_model_2)

            data_1 = data_1.to(self.device)
            target_1 = target_1.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data_1)
            loss = self.criterion(output, target_1)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.eq(target_1.data).cpu().sum()
            accuracy = 100. * correct.to(torch.float32) / len(data_1)
            accuracies.append(accuracy.data.item())
            losses = loss.data.item()

            data_2 = data_2.to(self.device)
            target_2 = target_2.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data_2)
            loss = self.criterion(output, target_2)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.eq(target_2.data).cpu().sum()
            accuracy = 100. * correct.to(torch.float32) / len(data_2)
            accuracies.append(accuracy.data.item())
            losses = loss.data.item()
        return losses, accuracies

    @staticmethod
    def free_memory():
        torch.cuda.empty_cache()
        del variables
        gc.collect()
