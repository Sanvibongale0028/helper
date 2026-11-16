import pandas as pd

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = [float(str(x).replace("%", "")) for x in df["Cocoa Percent"]]

pts = df[["Review Date", "Cocoa_Num", "Rating"]].iloc[0:5].values.tolist()
n = len(pts)

# 1. Raw Centroid & Distances
cent_raw = []
for j in range(3):
    col_sum = 0.0
    for i in range(n):
        col_sum += pts[i][j]
    cent_raw.append(col_sum / n)

print("Raw Centroid:", [round(c, 4) for c in cent_raw])
for i in range(n):
    sq_sum, abs_sum, cube_sum = 0.0, 0.0, 0.0
    for j in range(3):
        diff = pts[i][j] - cent_raw[j]
        if diff < 0: diff = -diff
        sq_sum += diff ** 2
        abs_sum += diff
        cube_sum += diff ** 3
    print(f"Row {i+1} (Raw) -> Euc={sq_sum**0.5:.4f}, Man={abs_sum:.4f}, Mink={cube_sum**(1/3):.4f}")

# 2. Manual Min-Max Normalisation to [0, 1]
mins, maxs = [], []
for j in range(3):
    mn, mx = pts[0][j], pts[0][j]
    for i in range(n):
        if pts[i][j] < mn: mn = pts[i][j]
        if pts[i][j] > mx: mx = pts[i][j]
    mins.append(mn)
    maxs.append(mx)

pts_norm = [[(pts[i][j] - mins[j]) / (maxs[j] - mins[j]) for j in range(3)] for i in range(n)]

# 3. Normalised Centroid & Distances
cent_norm = []
for j in range(3):
    col_sum = 0.0
    for i in range(n):
        col_sum += pts_norm[i][j]
    cent_norm.append(col_sum / n)

print("\nNormalised Centroid:", [round(c, 4) for c in cent_norm])
for i in range(n):
    sq_sum, abs_sum, cube_sum = 0.0, 0.0, 0.0
    for j in range(3):
        diff = pts_norm[i][j] - cent_norm[j]
        if diff < 0: diff = -diff
        sq_sum += diff ** 2
        abs_sum += diff
        cube_sum += diff ** 3
    print(f"Row {i+1} (Norm) -> Euc={sq_sum**0.5:.4f}, Man={abs_sum:.4f}, Mink={cube_sum**(1/3):.4f}")