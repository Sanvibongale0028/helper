
import csv
import math
import matplotlib.pyplot as plt

# ------------------------------------------
# PART A: DATA TYPES AND VISUALIZATION
# ------------------------------------------

filename = "2019.csv"

# Read the CSV file
data = []

with open(filename, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        record = {
            "Country": row["Country or region"],
            "H_Score": float(row["Score"]),
            "GDP": float(row["GDP per capita"]),
            "Social_Support": float(row["Social support"]),
            "Life_Expectancy": float(row["Healthy life expectancy"]),
            "Freedom": float(row["Freedom to make life choices"])
        }
        data.append(record)

attributes = [
    "H_Score",
    "GDP",
    "Social_Support",
    "Life_Expectancy",
    "Freedom"
]

factor_names = [
    "GDP",
    "Social_Support",
    "Life_Expectancy",
    "Freedom"
]

print("DATASET CHARACTERISTICS")
print("Number of rows:", len(data))
print("Number of attributes:", len(attributes))
print("Total selected values:", len(data) * len(attributes))

# Attribute types
print("\nATTRIBUTE TYPES")
print("Country: Categorical")
for attr in attributes:
    print(attr + ": Numerical")

# Check missing values in the selected attributes
print("\nMISSING VALUES")
for attr in attributes:
    missing = 0
    for row in data:
        if row[attr] is None:
            missing += 1
    print(attr, ":", missing)

# Calculate minimum, maximum and range
print("\nATTRIBUTE RANGES")
for attr in attributes:
    values = [row[attr] for row in data]
    minimum = min(values)
    maximum = max(values)

    print(attr)
    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("Range:", maximum - minimum)

# Create happiness classes using score tertiles
scores = sorted([row["H_Score"] for row in data])
n = len(scores)

low_cut = scores[n // 3]
high_cut = scores[(2 * n) // 3]

for row in data:
    if row["H_Score"] <= low_cut:
        row["Class"] = "Low"
    elif row["H_Score"] <= high_cut:
        row["Class"] = "Medium"
    else:
        row["Class"] = "High"

print("\nHAPPINESS CLASS COUNTS")
class_names = ["Low", "Medium", "High"]
class_counts = {}

for name in class_names:
    count = 0
    for row in data:
        if row["Class"] == name:
            count += 1
    class_counts[name] = count
    print(name, ":", count)

# 1. Histograms of all five attributes
for attr in attributes:
    values = [row[attr] for row in data]

    plt.figure()
    plt.hist(values, bins=10, edgecolor="black")
    plt.title("Histogram of " + attr)
    plt.xlabel(attr)
    plt.ylabel("Frequency")
    plt.tight_layout()

plt.show()

# 2. Scatter plots of each factor versus H_Score
for factor in factor_names:
    x = [row[factor] for row in data]
    y = [row["H_Score"] for row in data]

    plt.figure()
    plt.scatter(x, y)
    plt.title(factor + " vs H_Score")
    plt.xlabel(factor)
    plt.ylabel("H_Score")
    plt.tight_layout()

plt.show()

# 3. Mean factor values for each happiness class
means = {}

for name in class_names:
    class_rows = []

    for row in data:
        if row["Class"] == name:
            class_rows.append(row)

    means[name] = {}

    for factor in factor_names:
        total = 0

        for row in class_rows:
            total += row[factor]

        if len(class_rows) > 0:
            means[name][factor] = total / len(class_rows)
        else:
            means[name][factor] = 0

print("\nMEAN FACTOR VALUES BY CLASS")
for name in class_names:
    print(name, means[name])

# Grouped bar chart
positions = list(range(len(factor_names)))
width = 0.25

plt.figure()

for class_index, name in enumerate(class_names):
    values = [means[name][factor] for factor in factor_names]

    shifted_positions = [
        pos + (class_index - 1) * width
        for pos in positions
    ]

    plt.bar(
        shifted_positions,
        values,
        width=width,
        label=name
    )

plt.xticks(positions, factor_names, rotation=20)
plt.xlabel("Factors")
plt.ylabel("Mean value")
plt.title("Mean Factors by Happiness Class")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Pie chart of happiness class counts
pie_values = [class_counts[name] for name in class_names]

plt.figure()
plt.pie(
    pie_values,
    labels=class_names,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Happiness Class Distribution")
plt.tight_layout()
plt.show()
