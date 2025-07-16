
"""
Train script for sentiment analysis using a YAML config file.
"""
import argparse
import yaml
import random
from datasets import load_dataset
from torch.utils.data import DataLoader
from data4allnlp.data.sentiment_dataset import SentimentAnalysisDataset
from data4allnlp.models._registry import create_model_and_tokenizer
from data4allnlp.train.callback import EarlyStopping, LossHistoryLogger
from data4allnlp.train.trainer import Trainer
from data4allnlp.train import create_optimizer, create_scheduler, create_loss

def get_balanced_samples(dataset_split, n_per_class=20):
    pos = [item for item in dataset_split if item["label"] == 1]
    neg = [item for item in dataset_split if item["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)
    pos = pos[:n_per_class]
    neg = neg[:n_per_class]
    samples = pos + neg
    random.shuffle(samples)
    return samples

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model with YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Load dataset
    dataset = load_dataset(config["dataset_name"])
    train_samples = get_balanced_samples(dataset[config["train_split"]], n_per_class=config["n_per_class"])
    val_samples = get_balanced_samples(dataset[config["val_split"]], n_per_class=config["n_per_class_val"])

    # 2. Model and tokenizer
    model, tokenizer = create_model_and_tokenizer(
        config["model_id"],
        config["num_labels"]
    )

    # 3. Dataset and DataLoader
    train_dataset = SentimentAnalysisDataset(
        train_samples, tokenizer, max_length=config.get("max_length", 128)
    )
    val_dataset = SentimentAnalysisDataset(
        val_samples, tokenizer, max_length=config.get("max_length", 128)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.get("batch_size", 16), shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get("batch_size", 16)
    )

    # 4. Loss, optimizer, scheduler
    loss_fn = create_loss(config.get("loss_name", "cross_entropy"))
    optimizer = create_optimizer(
        model,
        optimizer_name=config.get("optimizer_name", "adamw"),
        lr=config.get("lr", 5e-5)
    )
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = create_scheduler(
            optimizer,
            scheduler_name=config.get("scheduler_name", "steplr"),
            step_size=config.get("step_size", 2),
            gamma=config.get("gamma", 0.8)
        )

    # 5. Callbacks
    callbacks = []
    if config.get("use_early_stopping", True):
        callbacks.append(EarlyStopping(
            patience=config.get("early_stopping_patience", 10),
            min_delta=config.get("early_stopping_min_delta", 0.01)
        ))
    callbacks.append(LossHistoryLogger())

    # 6. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=config.get("device", "auto"),
        output_dir=config.get("output_dir", "./output"),
        save_every=config.get("save_every", 10),
        log_every=config.get("log_every", 10),
        validate_every=config.get("validate_every", 50),
        max_epochs=config.get("max_epochs", 4),
        callbacks=callbacks
    )

    # 7. Run training
    trainer.train()

if __name__ == "__main__":
    main()