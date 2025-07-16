import torch
from typing import List, Union, Optional
import json
from data4allnlp.models import create_model_and_tokenizer
from data4allnlp.utils.device import select_device

class SentimentPredictor:
    """
    Class for performing inference on single or multiple texts using a fine-tuned HuggingFace model.

    Args:
        model_id (str): HuggingFace model id or path.
        num_labels (int): Number of output classes.
        weights_path (Optional[str]): Path to fine-tuned model weights (state_dict or full model).
        device (str): Device to use ("auto", "cpu", "cuda", "mps").
    """
    def __init__(
        self,
        model_id: str,
        num_labels: int,
        weights_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.model, self.tokenizer = create_model_and_tokenizer(
            model_id=model_id, 
            num_labels=num_labels,
            weights_path=weights_path,)
        self.device = select_device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        save_json: Optional[str] = None,
        batch_size: int = 8,
    ) -> Union[dict, List[dict]]:
        """
        Predict sentiment for a single text or a list of texts.

        Args:
            texts (str or List[str]): Text(s) to classify.
            save_json (Optional[str]): If set, save results to this file (one JSON per text if list, or single JSON if str).
            batch_size (int): Batch size for inference.

        Returns:
            dict or List[dict]: Prediction(s) with label and score.
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            for j, text in enumerate(batch_texts):
                result = {
                    "text": text,
                    "predicted_label": preds[j].item(),
                    "probabilities": probs[j].cpu().tolist(),
                }
                results.append(result)

        # Save results if requested
        if save_json:
            if single_input:
                with open(save_json, "w") as f:
                    json.dump(results[0], f, indent=2)
            else:
                for idx, res in enumerate(results):
                    with open(f"{save_json.rstrip('.json')}_{idx}.json", "w") as f:
                        json.dump(res, f, indent=2)

        return results[0] if single_input else results