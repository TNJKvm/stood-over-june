import torch
from torch import nn

from torch.nn import functional as F
from src.utils.losses import compute_irreducable_loss_batch


def irreducible_loss_selection(batch, target, selected_batch_size, ir_model, target_modell,
                               device=torch.device('cuda')):
    if next(ir_model.parameters()).is_cuda:
        batch = batch.to(device)
        target = target.to(device)
    irr_losses = compute_irreducable_loss_batch(batch, target, ir_model)

    sorted_idx = torch.argsort(irr_losses)
    selected_batch = sorted_idx[:selected_batch_size]

    return batch[selected_batch], target[selected_batch], selected_batch


def uniform_selection(batch, target, selected_batch_size, ir_model, target_modell, device=torch.device('cuda')):
    selected_batch = torch.randperm(len(batch))[:selected_batch_size]
    return batch[selected_batch], target[selected_batch], selected_batch


def reducible_loss_selection(batch, target, selected_batch_size, ir_model, target_modell: nn.Module,
                             device=torch.device('cuda')):
    if next(target_modell.parameters()).is_cuda:
        batch = batch.to(device)
        target = target.to(device)
    model_loss = F.cross_entropy(
        get_logits(target_modell(batch)), target, reduction="none"
    )
    irr_losses = compute_irreducable_loss_batch(batch, target, ir_model)

    reducible_loss = model_loss - irr_losses

    sorted_idx = torch.argsort(reducible_loss, descending=True)
    selected_batch = sorted_idx[:selected_batch_size]

    return batch[selected_batch], target[selected_batch], selected_batch


def loss_selection(batch, target, selected_batch_size, ir_model, target_modell, device=torch.device('cuda')):
    if next(target_modell.parameters()).is_cuda:
        batch = batch.to(device)
        target = target.to(device)
    model_loss = F.cross_entropy(
        get_logits(target_modell(batch)), target, reduction="none"
    )

    sorted_idx = torch.argsort(model_loss, descending=True)
    selected_batch = sorted_idx[:selected_batch_size]

    return batch[selected_batch], target[selected_batch], selected_batch


def grad_norm_selection(batch, target, selected_batch_size, ir_model, target_modell, device=torch.device('cuda')):
    with torch.inference_mode():
        if next(target_modell.parameters()).is_cuda:
            batch = batch.to(device)
            target = target.to(device)
        logits = get_logits(target_modell(batch))
        _, num_classes = logits.shape
        probs = F.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(target, num_classes=num_classes)
        g_i_norm_ub = torch.norm(probs - one_hot_targets, dim=-1)

    sorted_idx = torch.argsort(g_i_norm_ub, descending=True)
    selected_batch = sorted_idx[:selected_batch_size]

    return batch[selected_batch], target[selected_batch], selected_batch


def get_logits(input):
    if str(type(input)) == "<class 'torchvision.models.googlenet.GoogLeNetOutputs'>":
        return input.logits
    else:
        return input
