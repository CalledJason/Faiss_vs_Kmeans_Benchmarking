import time
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

def generate_benchmark_charts() -> None:
    df = pd.read_csv("results/benchmark_summary.csv")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df["Num_Samples"]))
    width = 0.35
    
    # Execution time
    axes[0].bar(x - width/2, df["Sklearn_Time_Sec"], width, label="Scikit-Learn K-Means", color="#4C72b0")
    axes[0].bar(x + width/2, df["FAISS_Time_Sec"], width, label="Meta FAISS K-Means", color="#DD8452")
    axes[0].set_title("Clustering Execution Time (Lower = Faster)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{n:,} rows" for n in df["Num_Samples"]])
    axes[0].set_ylabel("Time (Seconds)")
    axes[0].legend()

    # Clustering quality (Adjusted Rand Index)
    axes[1].plot(df["Num_Samples"].astype(str), df["ARI_Sklearn"], marker="o", linewidth=2.5, label="Scikit-Learn ARI")
    axes[1].plot(df["Num_Samples"].astype(str), df["ARI_FAISS"], marker="s", linewidth=2.5, linestyle="--", label="Meta FAISS ARI")
    axes[1].set_title("Topic Clustering Accuracy (ARI Score)", fontweight="bold")
    axes[1].set_xlabel("Dataset Size")
    axes[1].set_ylabel("Adjusted Rand Index (0 - 1)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    output_img = "results/benchmark_comparison.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Clustering benchmark chart saved to: {output_img}")

def evaluate_ivf_tradeoff(X: np.ndarray, df: pd.DataFrame) -> None:
    dim = X.shape[1]
    k_neighbors = 10

    # Ground Truth: Exact Brute-Force Search (IndexFlatIP)
    exact_index = faiss.IndexFlatIP(dim)
    exact_index.add(X)

    # Use 100 random queries from the dataset to measure statistically reliable latency & recall
    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(X), size=100, replace=False)
    queries = X[query_indices]

    t0 = time.perf_counter()
    _, exact_neighbors = exact_index.search(queries, k_neighbors)
    exact_latency_ms = ((time.perf_counter() - t0) / len(queries)) * 1000

    # Approximate Search: IVF Index across different nprobe values
    nlist = 50
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(X)
    index_ivf.add(X)

    nprobe_values = [1, 5, 15, 50]
    tradeoff_records = []

    print(f"\n=== FAISS IVF RECALL vs. LATENCY TRADE-OFF (100 Queries, Top-{k_neighbors}) ===")
    print(f"Exact Flat Search Baseline Latency: {exact_latency_ms:.4f} ms/query (Recall@10: 100.00%)\n")

    for nprobe in nprobe_values:
        index_ivf.nprobe = nprobe
        t0 = time.perf_counter()
        _, ivf_neighbors = index_ivf.search(queries, k_neighbors)
        avg_latency_ms = ((time.perf_counter() - t0) / len(queries)) * 1000

        # Compute Recall@10: fraction of true top-10 neighbors recovered by IVF
        recalls = [
            len(set(exact_neighbors[i]).intersection(set(ivf_neighbors[i]))) / k_neighbors
            for i in range(len(queries))
        ]
        mean_recall = np.mean(recalls) * 100

        tradeoff_records.append({
            "nprobe": nprobe,
            "Avg_Latency_ms": round(avg_latency_ms, 4),
            "Speedup_vs_Exact": round(exact_latency_ms / avg_latency_ms, 2),
            "Recall_at_10_Pct": round(mean_recall, 2)
        })

    tradeoff_df = pd.DataFrame(tradeoff_records)
    tradeoff_df.to_csv("results/ivf_tradeoff_summary.csv", index=False)
    print(tradeoff_df.to_string(index=False))

    # Plot Recall vs. Latency Trade-off Curve
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    sns.set_theme(style="whitegrid")

    color_recall = "#2E8B57"
    color_latency = "#D95F02"

    ax1.set_xlabel("nprobe (Number of Voronoi Cells Searched out of 50)")
    ax1.set_ylabel("Recall@10 (%)", color=color_recall, fontweight="bold")
    ax1.plot(tradeoff_df["nprobe"].astype(str), tradeoff_df["Recall_at_10_Pct"], marker="o", linewidth=2.5, color=color_recall, label="Recall@10 (%)")
    ax1.tick_params(axis="y", labelcolor=color_recall)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg Query Latency (ms)", color=color_latency, fontweight="bold")
    ax2.plot(tradeoff_df["nprobe"].astype(str), tradeoff_df["Avg_Latency_ms"], marker="s", linestyle="--", linewidth=2.5, color=color_latency, label="Latency (ms)")
    ax2.tick_params(axis="y", labelcolor=color_latency)

    plt.title("FAISS IndexIVFFlat: Accuracy (Recall@10) vs. Latency Trade-off", fontweight="bold")
    fig.tight_layout()
    plt.savefig("results/ivf_tradeoff.png", dpi=300)
    plt.close()
    print("\nTrade-off chart saved to: results/ivf_tradeoff.png")

    # Qualitative Semantic Search Demo (using balanced nprobe=5)
    index_ivf.nprobe = 5
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "NASA launches new space telescope to study distant galaxies"
    query_vec = model.encode([query_text], normalize_embeddings=True).astype("float32")

    t0 = time.perf_counter()
    scores, indices = index_ivf.search(query_vec, k=3)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n=== QUALITATIVE SEMANTIC SEARCH DEMO (nprobe=5) ===")
    print(f"Query: '{query_text}' | Latency: {latency_ms:.3f} ms\n")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"#{rank} [Cosine Sim: {score:.4f}] -> {df.iloc[idx]['text'][:120]}...")

if __name__ == "__main__":
    generate_benchmark_charts()
    df_news = pd.read_parquet("data/ag_news_subset.parquet")
    embeddings = np.load("data/embeddings.npy")
    evaluate_ivf_tradeoff(embeddings, df_news)