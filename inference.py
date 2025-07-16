
import argparse
import yaml
import json
from data4allnlp.inference.predictor import SentimentPredictor

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Run sentiment inference on input text using a YAML config.")
    parser.add_argument("--config", type=str, required=False, default="config/config_inference.yaml", help="Path to YAML config file for inference.")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save the result JSON.")
    args = parser.parse_args()

    config = load_config(args.config)

    predictor = SentimentPredictor(
        model_id=config["model_id"],
        num_labels=config["num_labels"],
        weights_path=config.get("weights_path"),
        device=config.get("device", "auto"),
    )

    result = predictor.predict(args.text, save_json=args.save_json)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()