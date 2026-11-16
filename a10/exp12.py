import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
cols = ["Economy", "Fam", "Health", "Freedom", "H_Score"]
points = df[cols].iloc[50:56].values.tolist()  # Rows 51-56 (0-based index 50 to 55)
n = len(points)
d = len(cols)

# Compute centroid
centroid = [sum(points[i][j] for i in range(n)) / n for j in range(d)]
print("Centroid (Rows 51-56):", [round(c, 4) for c in centroid], "\n")

# Distances from centroid
euc_dists, man_dists, min_dists = [], [], []
for i, p in enumerate(points):
    euc = sum((p[j] - centroid[j]) ** 2 for j in range(d)) ** (1 / 2)
    man = sum(abs(p[j] - centroid[j]) for j in range(d))
    mink = sum(abs(p[j] - centroid[j]) ** 3 for j in range(d)) ** (1 / 3)
    euc_dists.append(euc)
    man_dists.append(man)
    min_dists.append(mink)
    print(f"Row {51+i}: Euclidean={euc:.4f}, Manhattan={man:.4f}, Minkowski(p=3)={mink:.4f}")

# Lower triangular Euclidean matrix
print("\nLower Triangular Euclidean Distance Matrix:")
for i in range(n):
    row_str = []
    for j in range(i + 1):
        dist = sum((points[i][k] - points[j][k]) ** 2 for k in range(d)) ** 0.5
        row_str.append(f"{dist:.4f}")
    print(f"Row {51+i}: " + "\t".join(row_str))

# Nearest and farthest comparison
print("\nComparison of Measures:")
near_set, far_set = set(), set()
for name, dists in [("Euclidean", euc_dists), ("Manhattan", man_dists), ("Minkowski(p=3)", min_dists)]:
    near_row = 51 + dists.index(min(dists))
    far_row = 51 + dists.index(max(dists))
    near_set.add(near_row)
    far_set.add(far_row)
    print(f"{name}: Nearest = Row {near_row}, Farthest = Row {far_row}")

print("Do all three measures agree?", "Yes" if (len(near_set) == 1 and len(far_set) == 1) else "No")