# from datasets import load_dataset
# from transformers import AutoTokenizer
# from data4allnlp.data.sentiment_dataset import SentimentAnalysisDataset

# # Carica IMDB
# dataset = load_dataset("imdb")

# # Prendi una porzione del dataset per il train
# sample_data = [
#     {"text": dataset["train"][i]["text"], "label": dataset["train"][i]["label"]}
#     for i in range(5)
# ]

# # Inizializza il tokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# # crea il dataset per il sentiment analysis
# dataset = SentimentAnalysisDataset(
#     data = sample_data, 
#     tokenizer = tokenizer, 
#     max_length=128
# )

# # Crea il dataset PyTorch
# print(dataset[0])



def test_create_model_and_tokenizer():
    from data4allnlp.models._registry import create_model_and_tokenizer

    model_id = "distilbert-base-uncased"
    num_labels = 2
    try:
        model, tokenizer = create_model_and_tokenizer(model_id, num_labels)
        assert model is not None
        assert tokenizer is not None
        print("Model and tokenizer loaded successfully!")
        print("Model:", type(model))
        print("Tokenizer:", type(tokenizer))
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_create_model_and_tokenizer()