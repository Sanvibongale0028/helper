import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
features = ["Economy", "Fam", "Health", "Freedom"]
X = df[features]

# Apply K-Means with k = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# 1. Print cluster centres
centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("Cluster Centres:\n", centres, "\n")

# 2. Print cluster sizes
print("Cluster Sizes:\n", df["Cluster"].value_counts().sort_index(), "\n")

# 3. Compute Silhouette Coefficient
sil_score = silhouette_score(X, df["Cluster"])
print(f"Silhouette Coefficient: {sil_score:.4f}\n")

# 4. Compare mean H_Score of each cluster
print("Mean H_Score by Cluster:\n", df.groupby("Cluster")["H_Score"].mean())

# 5. Plot Economy vs Health coloured by cluster
plt.figure(figsize=(7, 5))
for c in range(3):
    subset = df[df["Cluster"] == c]
    plt.scatter(subset["Economy"], subset["Health"], label=f"Cluster {c}")

plt.xlabel("Economy")
plt.ylabel("Health")
plt.title("Economy vs Health by Cluster (k=3)")
plt.legend()
plt.show()