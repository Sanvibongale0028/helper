import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
X = df[["Economy", "Fam", "Health", "Freedom"]].values.tolist()
n = len(X)
k = 3

# 1. Initialize 3 random cluster centres
random.seed(42)
centres = [X[i][:] for i in random.sample(range(n), k)]

# 2. Run K-Means loop
for _ in range(50):
    clusters = [[] for _ in range(k)]
    labels = []
    
    # Assign each point to nearest centre
    for idx, p in enumerate(X):
        best_c = 0
        best_dist = sum((p[d] - centres[0][d]) ** 2 for d in range(4)) ** 0.5
        for c in range(1, k):
            dist = sum((p[d] - centres[c][d]) ** 2 for d in range(4)) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_c = c
        clusters[best_c].append(idx)
        labels.append(best_c)
        
    # Update centres
    new_centres = []
    for c in range(k):
        new_centres.append([sum(X[i][d] for i in clusters[c]) / len(clusters[c]) for d in range(4)])
        
    if new_centres == centres:
        break
    centres = new_centres

print("Cluster Centres:")
for c in range(k):
    print(f"Cluster {c}:", [round(v, 4) for v in centres[c]])

print("\nCluster Sizes:", [len(c) for c in clusters])

# 3. Manual Silhouette Coefficient
sil_total = 0
for i in range(n):
    c_i = labels[i]
    same = [j for j in clusters[c_i] if j != i]
    a_i = sum(sum((X[i][d] - X[j][d]) ** 2 for d in range(4)) ** 0.5 for j in same) / len(same)
    
    b_i = 999999
    for c in range(k):
        if c != c_i:
            other = clusters[c]
            dist_c = sum(sum((X[i][d] - X[j][d]) ** 2 for d in range(4)) ** 0.5 for j in other) / len(other)
            if dist_c < b_i:
                b_i = dist_c
                
    sil_total += (b_i - a_i) / (a_i if a_i > b_i else b_i)

print("\nSilhouette Coefficient:", round(sil_total / n, 4))

# 4. Mean H_Score per cluster
print("\nMean H_Score per Cluster:")
for c in range(k):
    mean_h = sum(df["H_Score"].iloc[i] for i in clusters[c]) / len(clusters[c])
    print(f"Cluster {c}: {mean_h:.4f}")

# 5. Plot Economy vs Health coloured by cluster
plt.scatter(df["Economy"], df["Health"], c=labels)
plt.xlabel("Economy")
plt.ylabel("Health")
plt.show()