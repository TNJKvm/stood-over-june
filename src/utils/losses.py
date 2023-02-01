import logging as log
import torch
from torch import nn

from torch.nn import functional as F


def get_irreducable_losses_complete(dataloader, small_model):
    log.info('Calculating irreducible losses')
    irr_losses = []
    with torch.inference_mode():
        for idx, (data, target) in enumerate(dataloader):
            irr_losses.append(compute_irreducable_loss_batch(data, target, small_model))
    return irr_losses


def compute_irreducable_loss_batch(data, target, small_model):
    output = small_model(data)
    if str(type(output)) == "<class 'torchvision.models.googlenet.GoogLeNetOutputs'>":
        output = output.logits
    return F.cross_entropy(
        output, target, reduction="none"
    )


def compute_reducable_loss_batch(large_model: nn.Module, small_model: nn.Module, data, target):
    with torch.inference_mode():
        logits = large_model(data)
        if str(type(logits)) == "<class 'torchvision.models.googlenet.GoogLeNetOutputs'>":
            logits = logits.logits
        model_loss = F.cross_entropy(logits, target, reduction="none")
        irreducible_loss = compute_irreducable_loss_batch(data, target, small_model)
        reducible_loss = model_loss - irreducible_loss

        return model_loss, reducible_loss
