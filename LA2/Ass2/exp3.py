
import pandas as pd
import math

df = pd.read_csv("happiness.csv")
df.columns = df.columns.str.strip()

def find_column(names):
    for name in names:
        for col in df.columns:
            if col.lower().strip() == name.lower():
                return col
    for name in names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    raise ValueError("Column not found: " + str(names))

family_col = find_column(["Fam", "Family", "Social support"])
health_col = find_column(["Health", "Health (Life Expectancy)", "Healthy life expectancy"])

def population_std(values):
    mean_value = sum(values) / len(values)
    return math.sqrt(
        sum((x - mean_value) ** 2 for x in values) / len(values)
    )

for col in [family_col, health_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in [family_col, health_col]:
    values = df[col].dropna().tolist()

    minimum = min(values)
    maximum = max(values)
    mean_value = sum(values) / len(values)
    std_value = population_std(values)

    minmax = [
        (x - minimum) / (maximum - minimum)
        if maximum != minimum else 0.0
        for x in values
    ]

    zscore = [
        (x - mean_value) / std_value
        if std_value != 0 else 0.0
        for x in values
    ]

    print("\nATTRIBUTE:", col)

    print("\nFirst 10 values")
    sample = pd.DataFrame({
        "Original": values[:10],
        "Min-Max [0,1]": minmax[:10],
        "Z-score": zscore[:10]
    })
    print(sample.round(4).to_string(index=False))

    print("\nVerification statistics")
    for method_name, transformed in [
        ("Original", values),
        ("Min-Max", minmax),
        ("Z-score", zscore)
    ]:
        print(
            method_name,
            "| Mean:", round(sum(transformed) / len(transformed), 4),
            "| Std:", round(population_std(transformed), 4),
            "| Min:", round(min(transformed), 4),
            "| Max:", round(max(transformed), 4)
        )
