from typing import Any, Dict, Optional, Tuple
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import logging
import torch

logger = logging.getLogger("data4allnlp.model_registry")

def create_model_and_tokenizer(
    model_id: str,
    num_labels: int,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    weights_path: Optional[str] = None,  # <-- aggiunto
    device: Optional[str] = None,        # <-- opzionale, per caricare su device corretto
    **kwargs: Any,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads a HuggingFace model and tokenizer for sequence classification.
    Optionally loads fine-tuned weights from a local path.

    Args:
        model_id (str): Model identifier from HuggingFace Hub.
        num_labels (int): Number of output classes.
        cache_dir (Optional[str]): Directory to cache models.
        revision (Optional[str]): Model revision.
        trust_remote_code (bool): Allow custom code from HF repo.
        weights_path (Optional[str]): Path to fine-tuned model weights.
        device (Optional[str]): Device for loading weights.
        **kwargs: Additional kwargs for model config.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizerBase]: Model and tokenizer.

    Raises:
        RuntimeError: If model or tokenizer cannot be loaded.
    """
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            num_labels=num_labels,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to load config from '{model_id}': {e}")
        raise RuntimeError(f"Could not load config from '{model_id}': {e}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer from '{model_id}': {e}")
        raise RuntimeError(f"Could not load tokenizer from '{model_id}': {e}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            config=config,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        # Carica i pesi fine-tunati se specificato
        if weights_path:
            map_location = device if device else "cpu"
            state = torch.load(weights_path, map_location=map_location)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
    except Exception as e:
        logger.error(f"Failed to load model from '{model_id}': {e}")
        raise RuntimeError(f"Could not load model from '{model_id}': {e}")
    
    logger.info(f"Loaded model and tokenizer from '{model_id}' with {num_labels} labels.")
    return model, tokenizer