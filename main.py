import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs

# ==========================================
# ЗАВДАННЯ 1:
# ==========================================
X, _ = make_blobs(
    n_samples=600,
    centers=5,
    cluster_std=1.2,
    random_state=0
)

df = pd.DataFrame(X, columns=["income", "spending_score"])

plt.figure(figsize=(8, 5))
plt.scatter(df["income"], df["spending_score"], s=35, alpha=0.75, edgecolors="k", linewidths=0.3)
plt.title("Початковий розподіл точок (make_blobs)")
plt.xlabel("income")
plt.ylabel("spending_score")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# ==========================================
# ЗАВДАННЯ 2:
# ==========================================
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
df["segment"] = kmeans.fit_predict(df[["income", "spending_score"]])

plt.figure(figsize=(8, 5))
scatter_km = plt.scatter(
    df["income"],
    df["spending_score"],
    c=df["segment"],
    cmap="tab10",
    s=35,
    alpha=0.85,
    edgecolors="k",
    linewidths=0.3
)
plt.title("K-Means Segmentation (n_clusters=5)")
plt.xlabel("income")
plt.ylabel("spending_score")
plt.colorbar(scatter_km, label="segment")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# ==========================================
# ЗАВДАННЯ 3:
# ==========================================
dbscan = DBSCAN(eps=1.5, min_samples=10)
df["cluster_dbscan"] = dbscan.fit_predict(df[["income", "spending_score"]])

plt.figure(figsize=(8, 5))
scatter_db = plt.scatter(
    df["income"],
    df["spending_score"],
    c=df["cluster_dbscan"],
    cmap="tab10",
    s=35,
    alpha=0.85,
    edgecolors="k",
    linewidths=0.3
)
plt.title("DBSCAN Segmentation (eps=1.5, min_samples=10)")
plt.xlabel("income")
plt.ylabel("spending_score")
plt.colorbar(scatter_db, label="cluster_dbscan")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

print("Розподіл за сегментами (K-Means):")
print(df["segment"].value_counts().sort_index())
print("\nРозподіл за кластерами (DBSCAN, -1 — шум):")
print(df["cluster_dbscan"].value_counts().sort_index())


# ==========================================
# ЗАВДАННЯ 4:
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc1 = axes[0].scatter(
    df["income"],
    df["spending_score"],
    c=df["segment"],
    cmap="tab10",
    s=35,
    alpha=0.85,
    edgecolors="k",
    linewidths=0.3
)
axes[0].set_title("K-Means Segmentation")
axes[0].set_xlabel("income")
axes[0].set_ylabel("spending_score")
axes[0].grid(True, linestyle="--", alpha=0.4)
fig.colorbar(sc1, ax=axes[0], label="segment")

sc2 = axes[1].scatter(
    df["income"],
    df["spending_score"],
    c=df["cluster_dbscan"],
    cmap="tab10",
    s=35,
    alpha=0.85,
    edgecolors="k",
    linewidths=0.3
)
axes[1].set_title("DBSCAN Segmentation")
axes[1].set_xlabel("income")
axes[1].set_ylabel("spending_score")
axes[1].grid(True, linestyle="--", alpha=0.4)
fig.colorbar(sc2, ax=axes[1], label="cluster_dbscan")

plt.tight_layout()
output_filename = "client_segmentation_comparison.png"
plt.savefig(output_filename, dpi=150)
plt.show()
print(f"\nПідсумкове зображення збережено у файл: {output_filename}")