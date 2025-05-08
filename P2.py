import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FILE = Path("Boston.csv") 
df = pd.read_csv(FILE)
if "medv" in df.columns:
    X = df.drop(columns=["medv"])
else:
    X = df.copy()
print(f"Shape of data: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
k_range = range(2, 7)
sil_scores = {}

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil
    print(f"k={k}  Silhouette={sil:.4f}")
best_k = max(sil_scores, key=sil_scores.get)
print(f"\n  Best k by silhouette: {best_k}")
best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = best_km.fit_predict(X_scaled)
cluster_means = (
    pd.DataFrame(X_scaled, columns=X.columns)
      .assign(cluster=labels)
      .groupby("cluster")
      .mean()
)
centroids = pd.DataFrame(best_km.cluster_centers_, columns=X.columns)
print("\n=== Cluster mean (scaled features) ===")
print(cluster_means.round(3))
print("\n=== Centroid coordinates (scaled) ===")
print(centroids.round(3))

# centroids_orig = pd.DataFrame(
#     scaler.inverse_transform(centroids), columns=X.columns
# )
# cluster_means_orig = pd.DataFrame(
#     scaler.inverse_transform(cluster_means), columns=X.columns
# )
