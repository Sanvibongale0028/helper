
import pandas as pd
import math

df = pd.read_csv("flavors_of_cacao.csv")
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

origin_col = find_column(["Bean Origin", "Broad Bean Origin"])
date_col = find_column(["Review Date"])
cocoa_col = find_column(["Cocoa Percent"])
rating_col = find_column(["Rating"])

print("Original record count:", len(df))

# 1. Bean Origin: least frequent values before cleaning
print("\n10 least-frequent Bean Origin values BEFORE cleaning")
print(df[origin_col].value_counts(dropna=False).tail(10).sort_values())

# Clean text: trim spaces, standardise case, standardise missing entries
def clean_text(value):
    if pd.isna(value):
        return "Unknown"

    value = " ".join(str(value).strip().split())

    if value == "":
        return "Unknown"

    return value.title()

df[origin_col] = df[origin_col].apply(clean_text)

print("\n10 least-frequent Bean Origin values AFTER cleaning")
print(df[origin_col].value_counts().sort_values().head(10))

# 2. Handle missing Review Date using its mode
df[date_col] = pd.to_numeric(df[date_col], errors="coerce")
date_mode = df[date_col].mode()

if len(date_mode) > 0:
    df[date_col] = df[date_col].fillna(date_mode.iloc[0])

print("\nMissing Review Dates after filling:", df[date_col].isna().sum())

# 3. Convert Cocoa Percent to numeric
df[cocoa_col] = pd.to_numeric(
    df[cocoa_col].astype(str).str.replace("%", "", regex=False).str.strip(),
    errors="coerce"
)

print("\nMissing Cocoa Percent after conversion:", df[cocoa_col].isna().sum())

# Fill invalid cocoa values with the median
valid_cocoa = sorted(df[cocoa_col].dropna().tolist())

if valid_cocoa:
    n = len(valid_cocoa)
    if n % 2 == 1:
        cocoa_median = valid_cocoa[n // 2]
    else:
        cocoa_median = (
            valid_cocoa[n // 2 - 1] + valid_cocoa[n // 2]
        ) / 2

    df[cocoa_col] = df[cocoa_col].fillna(cocoa_median)

# 4. Smooth Cocoa Percent using 15 equal-width bins
cocoa_values = df[cocoa_col].tolist()
minimum = min(cocoa_values)
maximum = max(cocoa_values)
bin_width = (maximum - minimum) / 15 if maximum != minimum else 0

bins = [[] for _ in range(15)]
bin_indices = []

for value in cocoa_values:
    if bin_width == 0:
        index = 0
    else:
        index = int((value - minimum) / bin_width)
        if index > 14:
            index = 14

    bins[index].append(value)
    bin_indices.append(index)

bin_means = []

for values in bins:
    if values:
        bin_means.append(sum(values) / len(values))
    else:
        bin_means.append(None)

df["Cocoa Percent Smoothed"] = [
    bin_means[index] for index in bin_indices
]

print("\nCocoa Percent smoothing (first 15 records)")
print(df[[cocoa_col, "Cocoa Percent Smoothed"]].head(15).round(3))

# 5. Min-Max normalisation of Rating
df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
rating_min = df[rating_col].min()
rating_max = df[rating_col].max()

if rating_max != rating_min:
    df["Rating Normalized"] = (
        (df[rating_col] - rating_min) / (rating_max - rating_min)
    )
else:
    df["Rating Normalized"] = 0.0

print("\nRating normalisation (first 10 records)")
print(df[[rating_col, "Rating Normalized"]].head(10).round(4))

# 6. Drop attributes not useful for mining
# Keep a copy of the full cleaned data before dropping columns
cleaned_df = df.copy()

drop_candidates = [
    col for col in df.columns
    if col.lower().strip() in [
        "ref", "review", "specific bean origin or bar name"
    ]
]

df = df.drop(columns=drop_candidates, errors="ignore")

print("\nAttributes dropped:", drop_candidates)
print("Final record count:", len(df))
print("Final columns:")
print(df.columns.tolist())

print("\nFinal cleaned dataset preview")
print(df.head())
