import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]
X = df[factors].copy()

# 1. Treat 0 values as missing and impute with column median
print("Zero values count before imputation:\n", (X == 0).sum(), "\n")
X = X.replace(0, np.nan)
X = X.fillna(X.median())

# 2. Detect and cap outliers using the IQR rule
for col in factors:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    outliers = X[(X[col] < lower_fence) | (X[col] > upper_fence)][col].tolist()
    print(f"{col} -> Fences: [{lower_fence:.4f}, {upper_fence:.4f}], Outliers detected: {outliers}")
    X[col] = X[col].clip(lower=lower_fence, upper=upper_fence)

# 3. Apply Z-score normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reduce the 4 factors to 2 principal components (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

evr = pca.explained_variance_ratio_
print(f"\nExplained Variance Ratio: PC1 = {evr[0]:.4f}, PC2 = {evr[1]:.4f}")
print(f"Total Variance Retained: {sum(evr) * 100:.2f}%")