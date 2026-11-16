# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
# df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# agg = df.groupby("Company").agg(
#     mean_rating=("Rating", "mean"),
#     mean_cocoa=("Cocoa_Num", "mean"),
#     reviews=("Rating", "count")
# )
# agg = agg[agg["reviews"] >= 5].copy()

# X_scaled = StandardScaler().fit_transform(agg)
# km = KMeans(n_clusters=3, random_state=42, n_init=10)
# agg["Cluster"] = km.fit_predict(X_scaled)

# print("Silhouette Coefficient:", round(silhouette_score(X_scaled, agg["Cluster"]), 4))
# print("\nCluster Summary (Means):\n", agg.groupby("Cluster").mean().round(2))

# plt.scatter(agg["mean_cocoa"], agg["mean_rating"], c=agg["Cluster"])
# plt.xlabel("Mean Cocoa %")
# plt.ylabel("Mean Rating")
# plt.title("Company Clusters: Mean Cocoa % vs Mean Rating (k = 3)")
# plt.show()

