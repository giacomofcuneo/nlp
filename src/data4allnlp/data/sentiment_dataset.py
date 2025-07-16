import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizerBase

class SentimentAnalysisDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis tasks.

    Expects input data as a list of dicts or a HuggingFace Dataset object,
    each with at least 'text' and 'label' fields.

    Args:
        data (Union[List[Dict[str, Any]], Any]): The dataset, either a list of dicts or a HuggingFace Dataset.
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer to preprocess the text.
        max_length (int): Maximum sequence length for tokenization.
        truncation (bool): Whether to truncate sequences to max_length.
        padding (str): Padding strategy. One of:
            - 'max_length': pad to max_length
            - 'longest': pad to the longest sequence in the batch
            - 'do_not_pad': no padding
    """
    def __init__(
        self,
        data: Union[List[Dict[str, Any]], Any],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single item for sentiment analysis.

        Args:
            idx (int): Index of the item.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with input_ids, attention_mask, and label.
        """
        item = self.data[idx] if isinstance(self.data, list) else self.data[int(idx)]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }