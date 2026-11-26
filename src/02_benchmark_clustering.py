import os
import time
import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

os.makedirs("results", exist_ok=True)

def run_benchmark() -> None:
    df = pd.read_parquet("data/ag_news_subset.parquet")
    X_full = np.load("data/embeddings.npy")
    y_full = df["label"].values

    scales = [500, 10_000, 60_000]
    k_clusters = 4
    dim = X_full.shape[1]
    records = []

    for n in scales:
        print(f"\nTesting scale of data: {n:,} rows...")
        X = X_full[:n]
        y_true = y_full[:n]

        # SKlearn KMeans (Classical)
        t0 = time.perf_counter()
        sk_km = SklearnKMeans(n_clusters=k_clusters, random_state=42, n_init=5, max_iter=25)
        sk_labels = sk_km.fit_predict(X)
        sk_time = time.perf_counter() - t0
        sk_ari = adjusted_rand_score(y_true, sk_labels)

        # Meta Faiss KMeans (Modern-Spherical)
        t0 = time.perf_counter()
        faiss_km = faiss.Kmeans(
            d=dim,
            k=k_clusters,
            niter=25,
            nredo=5,
            seed=42,
            spherical=True,
            verbose=False
        )
        faiss_km.train(X)
        _, faiss_labels = faiss_km.index.search(X, 1)
        faiss_labels = faiss_labels.flatten()
        faiss_time = time.perf_counter() - t0
        faiss_ari = adjusted_rand_score(y_true, faiss_labels)

        # Silhouette score evaluation on a max sample of 5.000 rows to save computation time 
        eval_n = min(n, 5_000)
        sk_sil = silhouette_score(X[:eval_n], sk_labels[:eval_n])
        faiss_sil = silhouette_score(X[:eval_n], faiss_labels[:eval_n])

        records.append({
            "Num_Samples": n,
            "Sklearn_Time_Sec": round(sk_time, 4),
            "FAISS_Time_Sec": round(faiss_time, 4),
            "FAISS_Speedup": round(sk_time / faiss_time, 2),
            "ARI_Sklearn": round(sk_ari, 4),
            "ARI_FAISS": round(faiss_ari, 4),
            "Silhouette_Sklearn": round(sk_sil, 4),
            "Silhouette_FAISS": round(faiss_sil, 4)
        })

    results_df = pd.DataFrame(records)
    output_csv = "results/benchmark_summary.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n=== Benchmark Result (Saved to {output_csv}) ===")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()