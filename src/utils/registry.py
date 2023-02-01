from torch import nn


class Model:
    def __init__(self, model: nn.Module, loss, validation_loss=None, validation_accuracy=None):
        self.model = model
        self.trainins_loss = loss
        self.validation_loss = validation_loss
        self.validation_accuracy = validation_accuracy
        self.model_number = None


class ModelRegistry:
    def __init__(self):
        self.registry = []

    def add_model(self, model: Model):
        model.model_number = len(self.registry)
        self.registry.append(model)

    def get_best_val_loss(self) -> Model:
        loss = self.registry[0].validation_loss
        best_idx = 0
        for idx, model in enumerate(self.registry):
            if model.validation_loss < loss:
                best_idx = idx
        return self.registry[best_idx]

    def get_best_val_acc(self) -> Model:
        acc = self.registry[0].validation_accuracy
        best_idx = 0
        for idx, model in enumerate(self.registry):
            if model.validation_accuracy > acc:
                best_idx = idx
        return self.registry[best_idx]

    def set_last_val_loss_acc(self, validation_accuracy, validation_loss):
        self.registry[-1].validation_accuracy = validation_accuracy
        self.registry[-1].validation_loss = validation_loss
