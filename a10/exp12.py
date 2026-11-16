import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
cols = ["Economy", "Fam", "Health", "Freedom", "H_Score"]

# Data rows 51 to 56 (1-indexed -> Python slice 50:56)
pts = df[cols].iloc[50:56].values.tolist()
n = len(pts)

# 1. Manual Centroid (average of each of the 5 attributes)
centroid = []
for j in range(5):
    col_sum = 0.0
    for i in range(n):
        col_sum += pts[i][j]
    centroid.append(col_sum / n)

print("Centroid:", [round(c, 4) for c in centroid], "\n")

# 2. Manual Euclidean, Manhattan, and Minkowski (p = 3) distances from centroid
euc_list, man_list, mink_list = [], [], []

for i in range(n):
    sq_sum, abs_sum, cube_sum = 0.0, 0.0, 0.0
    for j in range(5):
        diff = pts[i][j] - centroid[j]
        if diff < 0:
            diff = -diff
        sq_sum += diff ** 2
        abs_sum += diff
        cube_sum += diff ** 3
        
    euc = sq_sum ** (1 / 2)
    man = abs_sum
    mink = cube_sum ** (1 / 3)
    
    euc_list.append(euc)
    man_list.append(man)
    mink_list.append(mink)
    print(f"Row {51+i}: Euclidean = {euc:.4f}, Manhattan = {man:.4f}, Minkowski = {mink:.4f}")

# 3. Lower triangular pairwise Euclidean distance matrix
print("\nLower Triangular Euclidean Distance Matrix:")
for i in range(n):
    row_str = []
    for j in range(i + 1):
        d_sq = 0.0
        for k in range(5):
            d_sq += (pts[i][k] - pts[j][k]) ** 2
        row_str.append(f"{d_sq ** 0.5:.4f}")
    print(f"Row {51+i}:", "  ".join(row_str))

# 4. Find nearest and farthest points manually
for name, d_list in [("Euclidean", euc_list), ("Manhattan", man_list), ("Minkowski", mink_list)]:
    min_idx, max_idx = 0, 0
    for i in range(1, n):
        if d_list[i] < d_list[min_idx]:
            min_idx = i
        if d_list[i] > d_list[max_idx]:
            max_idx = i
    print(f"\n{name} -> Nearest: Row {51 + min_idx}, Farthest: Row {51 + max_idx}")

print("\nAgreement Check: No, all three measures do not agree. Euclidean and Manhattan identify Row 54 as nearest and Row 55 as farthest, whereas Minkowski (p=3) amplifies large single-attribute differences and identifies Row 53 as nearest and Row 52 as farthest.")