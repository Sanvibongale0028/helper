import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]
X = df[factors].copy()

# 1. Treat 0 values as missing and impute with column median
for col in factors:
    med = X[X[col] != 0][col].median()
    X[col] = X[col].replace(0, med)

# 2. Detect and cap outliers using the IQR rule
for col in factors:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    X[col] = X[col].clip(lower, upper)

# 3. Apply Z-score normalisation
X_scaled = StandardScaler().fit_transform(X)

# 4. Reduce the 4 factors to 2 principal components (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio (PC1, PC2):", pca.explained_variance_ratio_.round(4))
print("Total Variance Explained:", round(sum(pca.explained_variance_ratio_), 4))
print("\nFirst 5 Reduced Rows (PC1, PC2):\n", X_pca[:5].round(4))