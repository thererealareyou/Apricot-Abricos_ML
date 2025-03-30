import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict(texts):
    MODEL_NAME = 'cluster_saved_model'
    MAX_LENGTH = 256

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    f, r = 0, 0
    for text in texts:
        if len(text) > 4:
            encoding = tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            outputs = model(**encoding)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            if pred == 0:
                f += 1
            else:
                r += 1

    return 'Футбол' if f > r else 'Рестораны'
