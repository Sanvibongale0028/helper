
import pandas as pd
import math

df = pd.read_csv("weather.csv")
df.columns = df.columns.str.strip()

required = ["Outlook", "Temperature", "Humidity", "Windy", "Play"]

for col in required:
    if col not in df.columns:
        raise ValueError("Missing column: " + col)

df = df[required].dropna().copy()
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="raise")
df["Humidity"] = pd.to_numeric(df["Humidity"], errors="raise")
df["Play"] = df["Play"].astype(str).str.strip().str.title()

if len(df) != 14:
    print("Warning: Expected 14 records, found", len(df))

def entropy(data):
    if len(data) == 0:
        return 0

    counts = data["Play"].value_counts()
    total = len(data)
    result = 0

    for count in counts:
        p = count / total
        result -= p * math.log2(p)

    return result

def gini(data):
    if len(data) == 0:
        return 0

    counts = data["Play"].value_counts()
    total = len(data)
    result = 1

    for count in counts:
        p = count / total
        result -= p * p

    return result

parent_entropy = entropy(df)
parent_gini = gini(df)

print("Total records:", len(df))
print("Play counts:")
print(df["Play"].value_counts())
print("\nParent entropy:", round(parent_entropy, 4))
print("Parent Gini:", round(parent_gini, 4))

results = []

def evaluate_split(attribute, description, groups):
    total = len(df)

    weighted_entropy = sum(
        (len(group) / total) * entropy(group)
        for group in groups
    )

    weighted_gini = sum(
        (len(group) / total) * gini(group)
        for group in groups
    )

    information_gain = parent_entropy - weighted_entropy

    results.append({
        "Attribute": attribute,
        "Split": description,
        "Information Gain": information_gain,
        "Weighted Gini": weighted_gini
    })

    return information_gain, weighted_gini

# Categorical attribute: Outlook
for attribute in ["Outlook", "Windy"]:
    groups = [
        df[df[attribute] == value]
        for value in df[attribute].unique()
    ]

    evaluate_split(attribute, "Categorical split", groups)

# Numeric attributes: test every midpoint
numeric_working = {}

for attribute in ["Temperature", "Humidity"]:
    values = sorted(df[attribute].unique())
    working_rows = []

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2

        left = df[df[attribute] <= threshold]
        right = df[df[attribute] > threshold]
        groups = [left, right]

        ig, wg = evaluate_split(
            attribute,
            "<= " + str(threshold),
            groups
        )

        working_rows.append({
            "Threshold": threshold,
            "Left count": len(left),
            "Left Yes": (left["Play"] == "Yes").sum(),
            "Left No": (left["Play"] == "No").sum(),
            "Right count": len(right),
            "Right Yes": (right["Play"] == "Yes").sum(),
            "Right No": (right["Play"] == "No").sum(),
            "Left Entropy": entropy(left),
            "Right Entropy": entropy(right),
            "Information Gain": ig,
            "Left Gini": gini(left),
            "Right Gini": gini(right),
            "Weighted Gini": wg
        })

    numeric_working[attribute] = pd.DataFrame(working_rows)

# Full working table for Temperature
print("\nFULL WORKING TABLE: TEMPERATURE")
print(numeric_working["Temperature"].round(4).to_string(index=False))

print("\nFULL WORKING TABLE: HUMIDITY")
print(numeric_working["Humidity"].round(4).to_string(index=False))

# Compare all candidate splits
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Information Gain",
    ascending=False
)

print("\nALL SPLIT RESULTS")
print(results_df.round(4).to_string(index=False))

best_ig = max(results, key=lambda x: x["Information Gain"])
best_gini = min(results, key=lambda x: x["Weighted Gini"])

print("\nBest split by Information Gain:")
print(best_ig)

print("\nBest split by lowest Weighted Gini:")
print(best_gini)

if best_ig["Attribute"] == best_gini["Attribute"]:
    print("\nInformation Gain and Gini agree on the attribute.")
else:
    print("\nInformation Gain and Gini select different attributes.")
