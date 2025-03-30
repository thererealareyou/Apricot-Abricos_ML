import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


def train_model():
    MODEL_NAME = 'DeepPavlov/rubert-base-cased'
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    EPOCHS = 4
    LEARNING_RATE = 2e-5

    df = pd.read_csv('data/cluster_data.csv', sep='|')
    df = df.dropna()
    texts = df['text'].values
    labels = df['label'].values

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class CustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = int(self.labels[idx])

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)
            }

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}')

        model.eval()
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                val_predictions.extend(preds.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        print(classification_report(val_true_labels, val_predictions, target_names=['Футбол', 'Рестораны']))

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']. to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(classification_report(true_labels, predictions, target_names=['Футбол', 'Рестораны']))

    model.save_pretrained('cluster_saved_model')
    tokenizer.save_pretrained('cluster_saved_model')