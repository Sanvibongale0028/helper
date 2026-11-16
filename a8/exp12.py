import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
pts = df[["Economy", "Fam", "Health", "Freedom"]].iloc[0:6].values.tolist()  # First 6 rows
n = len(pts)

# 1. Calculate Centroid (average of each column)
centroid = [sum(pts[i][j] for i in range(n)) / n for j in range(4)]
print("Centroid:", [round(c, 4) for c in centroid], "\n")

# 2. Calculate Euclidean, Manhattan, and Minkowski (p=3) distances from centroid
euc_list, man_list, min_list = [], [], []

for i in range(n):
    euc = sum((pts[i][j] - centroid[j]) ** 2 for j in range(4)) ** (1/2)
    man = sum(abs(pts[i][j] - centroid[j]) for j in range(4))
    mink = sum(abs(pts[i][j] - centroid[j]) ** 3 for j in range(4)) ** (1/3)
    
    euc_list.append(euc)
    man_list.append(man)
    min_list.append(mink)
    print(f"Row {i+1}: Euclidean={euc:.4f}, Manhattan={man:.4f}, Minkowski={mink:.4f}")

# 3. Nearest and farthest point under each measure
for name, d_list in [("Euclidean", euc_list), ("Manhattan", man_list), ("Minkowski", min_list)]:
    near = d_list.index(min(d_list)) + 1
    far = d_list.index(max(d_list)) + 1
    print(f"\n{name} -> Nearest: Row {near}, Farthest: Row {far}")

# 4. Pairwise Euclidean distances (lower triangular matrix)
print("\nLower Triangular Euclidean Matrix:")
for i in range(n):
    row_vals = []
    for j in range(i + 1):
        dist = sum((pts[i][k] - pts[j][k]) ** 2 for k in range(4)) ** 0.5
        row_vals.append(f"{dist:.4f}")
    print(f"Row {i+1}:", "  ".join(row_vals))