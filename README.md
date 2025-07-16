# NLP Sentiment Analysis Library

This library provides tools for training and inference of sentiment analysis models using HuggingFace Transformers.

## Features

- Training with YAML configuration
- Inference with YAML configuration
- Early stopping, loss logging, and more

## Usage

See `train.py` and `inference.py` for examples.

## Requirements

- Python 3.8+
- torch
- transformers
- datasets
- pyyaml

## Quickstart

```sh
python [train.py](http://_vscodecontentref_/0) --config config.yaml
python [inference.py](http://_vscodecontentref_/1) --config config_inference.yaml --text "Your text here"