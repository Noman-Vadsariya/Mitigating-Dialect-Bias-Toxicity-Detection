import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

def get_embeddings(texts, batch_size=32):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokens)

        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def process_split(file_path, out_path):
    df = pd.read_csv(file_path)

    texts = df["tweet"].astype(str).tolist()
    X = get_embeddings(texts)

    np.save(out_path, X)

    print(f"Saved embeddings: {out_path}")


if __name__ == "__main__":
    process_split("../data/processed/train.csv", "../data/embeddings/train_emb.npy")
    process_split("../data/processed/val.csv", "../data/embeddings/val_emb.npy")
    process_split("../data/processed/test.csv", "../data/embeddings/test_emb.npy")