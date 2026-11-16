
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

location_col = find_column(["Company Location"])
cocoa_col = find_column(["Cocoa Percent"])
date_col = find_column(["Review Date"])
rating_col = find_column(["Rating"])

df[cocoa_col] = pd.to_numeric(
    df[cocoa_col].astype(str).str.replace("%", "", regex=False).str.strip(),
    errors="coerce"
)
df[date_col] = pd.to_numeric(df[date_col], errors="coerce")
df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

df = df.dropna(
    subset=[location_col, cocoa_col, date_col, rating_col]
).copy()

def rating_class(value):
    if value < 3.0:
        return "Low"
    elif value < 3.5:
        return "Medium"
    return "High"

df["Rating Class"] = df[rating_col].apply(rating_class)

# Company Location: top 5, all others become Other
top5 = df[location_col].value_counts().head(5).index.tolist()

df["Location Group"] = df[location_col].apply(
    lambda x: x if x in top5 else "Other"
)

def entropy(labels):
    counts = labels.value_counts()
    total = len(labels)
    result = 0

    for count in counts:
        p = count / total
        result -= p * math.log2(p)

    return result

def gini(labels):
    counts = labels.value_counts()
    total = len(labels)
    result = 1

    for count in counts:
        p = count / total
        result += 0
        result -= p * p

    return result

parent_entropy = entropy(df["Rating Class"])
parent_gini = gini(df["Rating Class"])

def evaluate_groups(attribute, description, groups):
    total = len(df)

    weighted_entropy = sum(
        (len(group) / total) * entropy(group["Rating Class"])
        for group in groups if len(group) > 0
    )

    weighted_gini = sum(
        (len(group) / total) * gini(group["Rating Class"])
        for group in groups if len(group) > 0
    )

    return {
        "Attribute": attribute,
        "Split": description,
        "Information Gain": parent_entropy - weighted_entropy,
        "Weighted Gini": weighted_gini
    }

results = []

# Categorical split: Company Location groups
location_groups = [
    group for _, group in df.groupby("Location Group")
]

results.append(
    evaluate_groups(
        "Company Location",
        "Top 5 locations + Other",
        location_groups
    )
)

# Numeric splits: every midpoint between consecutive distinct values
numeric_working = {}

for attribute in [cocoa_col, date_col]:
    unique_values = sorted(df[attribute].unique())
    rows = []

    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2

        left = df[df[attribute] <= threshold]
        right = df[df[attribute] > threshold]

        result = evaluate_groups(
            attribute,
            "<= " + str(threshold),
            [left, right]
        )

        results.append(result)

        rows.append({
            "Threshold": threshold,
            "Left records": len(left),
            "Left Low": (left["Rating Class"] == "Low").sum(),
            "Left Medium": (left["Rating Class"] == "Medium").sum(),
            "Left High": (left["Rating Class"] == "High").sum(),
            "Right records": len(right),
            "Right Low": (right["Rating Class"] == "Low").sum(),
            "Right Medium": (right["Rating Class"] == "Medium").sum(),
            "Right High": (right["Rating Class"] == "High").sum(),
            "Information Gain": result["Information Gain"],
            "Weighted Gini": result["Weighted Gini"]
        })

    numeric_working[attribute] = pd.DataFrame(rows)

print("Parent entropy:", round(parent_entropy, 4))
print("Parent Gini:", round(parent_gini, 4))

print("\nCOMPANY LOCATION SPLIT")
print(pd.DataFrame([results[0]]).round(5).to_string(index=False))

for attribute in [cocoa_col, date_col]:
    print("\nALL CANDIDATE THRESHOLDS FOR", attribute)
    print(numeric_working[attribute].round(5).to_string(index=False))

results_df = pd.DataFrame(results)

print("\nALL ATTRIBUTE AND THRESHOLD RESULTS")
print(
    results_df.sort_values(
        "Information Gain", ascending=False
    ).round(5).to_string(index=False)
)

best_ig = max(results, key=lambda x: x["Information Gain"])
best_gini = min(results, key=lambda x: x["Weighted Gini"])

print("\nBEST SPLIT BY INFORMATION GAIN")
print(best_ig)

print("\nBEST SPLIT BY LOWEST WEIGHTED GINI")
print(best_gini)

if best_ig["Attribute"] == best_gini["Attribute"]:
    print("\nBoth criteria select the same attribute.")
else:
    print("\nThe two criteria select different attributes.")
