import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

os.makedirs("data", exist_ok=True)

def prepare_data(max_samples: int = 60_000) -> None:
    print("Loading dataset AG News...")
    dataset = load_dataset("ag_news", split="train")

    df = pd.DataFrame(dataset).sample(n=max_samples, random_state=42).reset_index(drop=True)
    output_parquet = "data/ag-news_subset.parquet"
    df.to_parquet(output_parquet, index=False)
    print(f"Saved to {output_parquet} ({len(df):,} rows).")

    print("Extracting embeddings semantically (384 dimensions)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    output_npy = "data/embeddings.npy"
    np.save(output_npy, embeddings)
    print(f"Completed: Vector saved at {output_npy} | Shape: {embeddings.shape} ({embeddings.dtype})")

if __name__ == "__main__":
    prepare_data(max_samples=60_000)
