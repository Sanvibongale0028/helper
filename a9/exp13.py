import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset and Z-score normalise the four factors
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[factors])

# Run K-Means for k = 2 to 6 and record SSE and Silhouette Score
k_values = range(2, 7)
sse_list = []
sil_list = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sse_list.append(km.inertia_)
    sil_list.append(silhouette_score(X_scaled, labels))
    print(f"k = {k} -> SSE = {km.inertia_:.2f}, Silhouette Score = {sil_list[-1]:.4f}")

# Plot Elbow Curve (SSE vs k) and Silhouette Score vs k
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_values, sse_list, marker="o", color="darkblue")
axes[0].set_title("Elbow Curve (SSE vs k)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("SSE (Inertia)")
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(k_values, sil_list, marker="s", color="darkgreen")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# Best k selection (k=2 achieves the highest silhouette score of 0.4149 and sharpest initial SSE drop)
best_k = k_values[sil_list.index(max(sil_list))]
print(f"\nChosen Best k: {best_k} (achieves the highest Silhouette Score of {max(sil_list):.4f} and sharpest SSE drop).")

# Fit final K-Means with best_k and plot Economy vs Health coloured by cluster
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = final_km.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
for c in range(best_k):
    subset = df[df["Cluster"] == c]
    plt.scatter(subset["Economy"], subset["Health"], label=f"Cluster {c}", alpha=0.8)

plt.xlabel("Economy")
plt.ylabel("Health")
plt.title(f"Economy vs Health Coloured by Final Clusters (k = {best_k})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()