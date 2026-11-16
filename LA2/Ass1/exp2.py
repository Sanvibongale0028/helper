
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("happiness.csv")
df.columns = df.columns.str.strip()

def find_column(names):
    for name in names:
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    for name in names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    raise ValueError("Column not found: " + str(names))

score_col = find_column(["H_Score", "H Score", "Happiness Score", "Score"])
economy_col = find_column(["Economy", "Economy (GDP per Capita)", "GDP per capita"])

def median(values):
    values = sorted(values)
    n = len(values)

    if n == 0:
        return None
    if n % 2 == 1:
        return values[n // 2]

    return (values[n // 2 - 1] + values[n // 2]) / 2

def five_number_summary(values):
    values = sorted(values)
    n = len(values)

    if n == 0:
        raise ValueError("No valid numeric values.")

    med = median(values)

    if n % 2 == 0:
        lower_half = values[:n // 2]
        upper_half = values[n // 2:]
    else:
        lower_half = values[:n // 2]
        upper_half = values[n // 2 + 1:]

    q1 = median(lower_half)
    q3 = median(upper_half)

    return min(values), q1, med, q3, max(values)

for col in [score_col, economy_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    values = df[col].dropna().tolist()

    minimum, q1, med, q3, maximum = five_number_summary(values)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = [
        x for x in values
        if x < lower_fence or x > upper_fence
    ]

    print("\nAttribute:", col)
    print("Minimum:", minimum)
    print("Q1:", q1)
    print("Median:", med)
    print("Q3:", q3)
    print("Maximum:", maximum)
    print("IQR:", iqr)
    print("Lower fence:", lower_fence)
    print("Upper fence:", upper_fence)
    print("Outlier values:", sorted(outliers))

    plt.figure(figsize=(7, 4))
    plt.boxplot(values, vert=False)
    plt.title("Box Plot of " + col)
    plt.xlabel(col)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.show()
