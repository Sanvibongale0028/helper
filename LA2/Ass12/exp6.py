
import pandas as pd

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

score_col = find_column([
    "H Score", "Happiness Score", "Score", "N Score"
])

economy_col = find_column([
    "Economy", "Economy (GDP per Capita)", "GDP per capita"
])

df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df[economy_col] = pd.to_numeric(df[economy_col], errors="coerce")
df = df.dropna(subset=[score_col, economy_col]).copy()

# H Score concept hierarchy using fixed cut-offs
def score_level(score):
    if score >= 6.5:
        return "Very High"
    elif score >= 5.5:
        return "High"
    elif score >= 4.5:
        return "Medium"
    else:
        return "Low"

df["H Score Level"] = df[score_col].apply(score_level)

# Economy concept hierarchy using tertiles
economy_values = sorted(df[economy_col].tolist())
n = len(economy_values)

lower_index = n // 3
upper_index = (2 * n) // 3

lower_cutoff = economy_values[lower_index]
upper_cutoff = economy_values[upper_index]

def economy_level(value):
    if value < lower_cutoff:
        return "Poor"
    elif value < upper_cutoff:
        return "Middle"
    else:
        return "Rich"

df["Economy Level"] = df[economy_col].apply(economy_level)

print("Economy tertile cut-offs:")
print("Poor/Middle boundary:", lower_cutoff)
print("Middle/Rich boundary:", upper_cutoff)

# Mean of each factor for every H Score level
mean_table = df.groupby("H Score Level")[
    [score_col, economy_col]
].mean()

order = ["Very High", "High", "Medium", "Low"]
mean_table = mean_table.reindex(order)

print("\nMean of factors for each H Score level:")
print(mean_table.round(3))

# Economy level vs H Score level count table
count_table = pd.crosstab(
    df["Economy Level"],
    df["H Score Level"]
)

count_table = count_table.reindex(
    index=["Poor", "Middle", "Rich"],
    columns=order,
    fill_value=0
)

print("\nEconomy Level vs H Score Level:")
print(count_table)

print("\nTotal records:", len(df))
