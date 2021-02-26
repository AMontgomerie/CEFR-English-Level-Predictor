import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    Trainer,
    TrainingArguments,
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_CHECKPOINT = "bert-base-cased"
SAVE_DIR = "/content/drive/MyDrive/bert-cefr"
DATA_DIR = "/content/CEFR-English-Level-Predictor/data/cefr_leveled_texts.csv"
SEQ_LEN = 512

tokenizer = BertTokenizer.from_pretrained(MODEL_CHECKPOINT)


class CEFRDataset(Dataset):
    def __init__(self, texts, labels):
        encoder = LabelEncoder()
        self.texts = texts
        self.labels = encoder.fit_transform(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoded_text = tokenizer(
            text,
            padding="max_length",
            max_length=SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        encoded_text["input_ids"] = encoded_text["input_ids"].squeeze()
        encoded_text["attention_mask"] = encoded_text["attention_mask"].squeeze()
        label = torch.tensor(label)

        return {
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "labels": label,
        }

    def get_labels(self):
        return self.labels


def get_data(csv_path, train_size=0.8):
    data = pd.read_csv(csv_path)
    dataset = CEFRDataset(data["text"], data["label"])
    train_len = int(len(dataset) * train_size)
    valid_len = len(dataset) - train_len
    train, valid = random_split(dataset, [train_len, valid_len])
    print(f"{len(train)} training examples and {len(valid)} validation examples.")
    return train, valid


def train(train_set, valid_set, epochs=10, warmup_size=0.1, lr=1e-3, batch_size=16):
    model = get_model(MODEL_CHECKPOINT)
    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        optim, warmup_size, round(len(train_set) / batch_size * epochs)
    )
    training_args = get_training_args(epochs, batch_size)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        optimizers=[optim, scheduler],
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.save_model()


def get_model(pretrained_checkpoint):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_checkpoint, num_labels=6
    )
    return model.to(device)


def get_scheduler(optimizer, warmup_size, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warmup_size),
        num_training_steps=total_steps,
    )
    return scheduler


def get_training_args(epochs, batch_size):
    return TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=50,
        fp16=True,
        evaluation_strategy="epoch",
        eval_accumulation_steps=1,
    )


def compute_accuracy(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


if __name__ == "__main__":
    train_set, valid_set = get_data(DATA_DIR, train_size=0.8)
    train(train_set, valid_set, epochs=12, warmup_size=0.2, lr=2e-5, batch_size=16)