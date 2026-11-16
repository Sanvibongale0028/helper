import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# Aggregate to company level and filter companies with at least 5 reviews
comp_df = df.groupby("Company").agg(
    mean_Rating=("Rating", "mean"),
    mean_Cocoa=("Cocoa_Num", "mean"),
    num_reviews=("Rating", "count")
).reset_index()

comp_df = comp_df[comp_df["num_reviews"] >= 5].copy()
print(f"Total companies with >= 5 reviews: {len(comp_df)}\n")

# Normalise the aggregated features
features = ["mean_Rating", "mean_Cocoa", "num_reviews"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(comp_df[features])

# Apply K-Means with k = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
comp_df["Cluster"] = kmeans.fit_predict(X_scaled)

# Compute Silhouette Coefficient
sil = silhouette_score(X_scaled, comp_df["Cluster"])
print(f"Silhouette Coefficient: {sil:.4f}\n")

# Describe each cluster using original scale means & counts
cluster_summary = comp_df.groupby("Cluster")[features].agg(["count", "mean"])
print("=== Cluster Profiles ===")
print(cluster_summary, "\n")

print("Cluster Descriptions:")
print("- Cluster 0 (High-Rated Boutique Makers): High average rating (~3.42) with moderate review counts (~8 reviews).")
print("- Cluster 1 (Lower-Rated Small Makers): Lower average rating (~2.99) and fewer reviews (~6.6 reviews), slightly higher cocoa % (~71.9%).")
print("- Cluster 2 (High-Volume Established Makers): High review counts (~21.2 reviews on average) with solid ratings (~3.32).")

# Plot mean Cocoa % vs mean Rating coloured by cluster
plt.figure(figsize=(7, 5))
for c in range(3):
    sub = comp_df[comp_df["Cluster"] == c]
    plt.scatter(sub["mean_Cocoa"], sub["mean_Rating"], label=f"Cluster {c}", alpha=0.8)

plt.xlabel("Mean Cocoa %")
plt.ylabel("Mean Rating")
plt.title("Company Mean Cocoa % vs Mean Rating by Cluster (k=3)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()