from typing import Any, Optional

class Callback:
    """
    Base class for all callbacks.
    """
    def on_epoch_end(self, trainer: Any, epoch: int):
        pass

    def on_step_end(self, trainer, step: int, loss: float, mode: str):
        pass

    def on_validation_end(self, trainer: Any, val_loss: float, val_acc: float, step: int):
        pass

class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when validation loss does not improve.

    Args:
        patience (int): How many validations to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def on_validation_end(self, trainer: Any, val_loss: float, val_acc: float, step: int):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at step {step}.")
                self.should_stop = True

class LossHistoryLogger(Callback):
    """
    Callback to save loss history at the end of training.
    """
    def on_step_end(self, trainer, step: int, loss: float, mode: str):
        print(f"[Step {step}] {mode} loss: {loss:.4f}")