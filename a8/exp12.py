import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
cols = ["Economy", "Fam", "Health", "Freedom"]
points = df[cols].iloc[0:6].values.tolist()  # Rows 1-6 (index 0 to 5)
n = len(points)
d = len(cols)

# Compute centroid manually
centroid = [sum(points[i][j] for i in range(n)) / n for j in range(d)]
print("Centroid:", [round(c, 4) for c in centroid], "\n")

# Compute Euclidean, Manhattan, and Minkowski (p=3) distances from centroid
euc_dists, man_dists, min_dists = [], [], []
for i, p in enumerate(points):
    euc = sum((p[j] - centroid[j]) ** 2 for j in range(d)) ** (1 / 2)
    man = sum(abs(p[j] - centroid[j]) for j in range(d))
    mink = sum(abs(p[j] - centroid[j]) ** 3 for j in range(d)) ** (1 / 3)
    euc_dists.append(euc)
    man_dists.append(man)
    min_dists.append(mink)
    print(f"Row {i+1}: Euclidean={euc:.4f}, Manhattan={man:.4f}, Minkowski(p=3)={mink:.4f}")

# Identify nearest and farthest points under each measure
for name, dists in [("Euclidean", euc_dists), ("Manhattan", man_dists), ("Minkowski (p=3)", min_dists)]:
    near_idx = dists.index(min(dists)) + 1
    far_idx = dists.index(max(dists)) + 1
    print(f"\n{name} -> Nearest: Row {near_idx} ({min(dists):.4f}) | Farthest: Row {far_idx} ({max(dists):.4f})")

# Pairwise Euclidean distance lower triangular matrix
print("\nLower Triangular Pairwise Euclidean Distance Matrix:")
for i in range(n):
    row_str = []
    for j in range(i + 1):
        dist = sum((points[i][k] - points[j][k]) ** 2 for k in range(d)) ** 0.5
        row_str.append(f"{dist:.4f}")
    print(f"Row {i+1}: " + "\t".join(row_str))