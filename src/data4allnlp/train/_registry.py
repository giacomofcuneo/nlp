from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, _LRScheduler
import torch.nn as nn
from typing import Any, Dict

def create_optimizer(model, optimizer_name: str = "adamw", lr: float = 5e-5, **kwargs) -> Optimizer:
    """
    Returns an optimizer instance for the given model.
    """
    if optimizer_name.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_scheduler(optimizer: Optimizer, scheduler_name: str = "steplr", **kwargs) -> _LRScheduler:
    """
    Returns a learning rate scheduler instance.
    """
    if scheduler_name.lower() == "steplr":
        return StepLR(optimizer, step_size=kwargs.get("step_size", 1), gamma=kwargs.get("gamma", 0.1))
    elif scheduler_name.lower() == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def create_loss(loss_name: str = "cross_entropy", **kwargs) -> nn.Module:
    """
    Returns a loss function instance.
    """
    if loss_name.lower() == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name.lower() == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")