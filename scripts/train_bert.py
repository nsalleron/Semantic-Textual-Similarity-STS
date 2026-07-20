"""Fine-tune a BERT bi-encoder on STS Benchmark and save a reusable checkpoint.

Adapted from the project notebook (STSBertModel + CosineSimilarityLoss). The
trained model is a ``sentence-transformers`` model saved to
``artifacts/bert_finetuned/`` so it can be reloaded by the API like any other
Sentence-Transformers backend.

Usage:
    python -m scripts.train_bert                 # full training (8 epochs)
    python -m scripts.train_bert --subset 200 --epochs 1   # quick smoke run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from api.core.config import BERT_FINETUNED_DIR
from api.core.device import get_device

BASE_MODEL = "bert-base-uncased"
MAX_SEQ_LENGTH = 128


def build_model():
    """Build a SentenceTransformer (BERT transformer + mean pooling)."""
    from sentence_transformers import SentenceTransformer, models

    word_embedding_model = models.Transformer(BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH)
    # ``get_word_embedding_dimension`` was renamed to ``get_embedding_dimension``
    # in sentence-transformers 5.x; support both.
    get_dim = getattr(
        word_embedding_model,
        "get_embedding_dimension",
        getattr(word_embedding_model, "get_word_embedding_dimension", None),
    )
    pooling_model = models.Pooling(get_dim())
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


class DataSequence(torch.utils.data.Dataset):
    """STSB pairs tokenized for the bi-encoder, labels normalized to [0, 1]."""

    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        similarity = [row["similarity_score"] for row in dataset]
        self.label = [s / 5.0 for s in similarity]
        self.sentence_1 = [row["sentence1"] for row in dataset]
        self.sentence_2 = [row["sentence2"] for row in dataset]
        self.text_cat = [
            [str(x), str(y)] for x, y in zip(self.sentence_1, self.sentence_2)
        ]

    def __len__(self):
        return len(self.text_cat)

    def __getitem__(self, idx):
        texts = self.tokenizer(
            self.text_cat[idx],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor(self.label[idx])
        return texts, label


def collate_features(texts):
    """Split a batched tokenizer output into per-sentence feature dicts."""
    num_texts = len(texts["input_ids"])
    features = []
    for i in range(num_texts):
        features.append(
            {
                "input_ids": texts["input_ids"][i],
                "attention_mask": texts["attention_mask"][i],
            }
        )
    return features


class CosineSimilarityLoss(torch.nn.Module):
    """MSE between cosine similarity of the two embeddings and the gold label."""

    def __init__(self):
        super().__init__()
        self.loss_fct = torch.nn.MSELoss()
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, embeddings, label):
        embedding_1 = torch.stack([e[0] for e in embeddings])
        embedding_2 = torch.stack([e[1] for e in embeddings])
        output = self.cos(embedding_1, embedding_2)
        return self.loss_fct(output, label.squeeze())


def train(dataset, tokenizer, epochs: int, learning_rate: float, batch_size: int):
    device = get_device()
    print(f"Training on device: {device}")

    model = build_model().to(device)
    criterion = CosineSimilarityLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = DataSequence(dataset, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, num_workers=0, batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for train_data, train_label in tqdm(train_dataloader, desc=f"epoch {epoch+1}"):
            train_data["input_ids"] = train_data["input_ids"].to(device)
            train_data["attention_mask"] = train_data["attention_mask"].to(device)
            train_data.pop("token_type_ids", None)

            features = collate_features(train_data)
            output = [model(f)["sentence_embedding"] for f in features]

            loss = criterion(output, train_label.to(device))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} | mean loss: {total_loss / len(train_dataset):.4f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on STSB.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="If > 0, train on only the first N examples (quick smoke run).",
    )
    parser.add_argument("--output", type=str, default=str(BERT_FINETUNED_DIR))
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    dataset = load_dataset("PhilipMay/stsb_multi_mt", "en", split="train")
    if args.subset > 0:
        dataset = dataset.select(range(min(args.subset, len(dataset))))

    model = train(
        dataset,
        tokenizer,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    out_dir = Path(args.output)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
