
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("happiness.csv")
df.columns = [str(c).strip() for c in df.columns]


def find_column(candidates):
    for candidate in candidates:
        for col in df.columns:
            if col.lower().strip() == candidate.lower().strip():
                return col
    raise ValueError("Column not found: " + str(candidates))


economy_col = find_column(["Economy", "Economy (GDP per Capita)",
                           "Logged GDP per capita", "GDP per capita"])
family_col = find_column(["Family", "Social support"])
health_col = find_column(["Health", "Health (Life Expectancy)",
                          "Healthy life expectancy"])
freedom_col = find_column(["Freedom", "Freedom to make life choices"])

features = {
    "Economy": economy_col,
    "Family": family_col,
    "Health": health_col,
    "Freedom": freedom_col
}


def clean_values(column):
    values = []
    for value in df[column].tolist():
        try:
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        except (ValueError, TypeError):
            pass
    return values


def median(values):
    n = len(values)
    if n == 0:
        return None
    values = sorted(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = lower + 1
    if upper >= len(values):
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


box_data = []

print("FIVE-NUMBER SUMMARY AND OUTLIERS")

for name, column in features.items():
    values = clean_values(column)
    if not values:
        continue

    values.sort()
    minimum = values[0]
    maximum = values[-1]
    q1 = percentile(values, 0.25)
    med = median(values)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = []
    for value in values:
        if value < lower_limit or value > upper_limit:
            outliers.append(value)

    print("\n", name)
    print("Minimum:", round(minimum, 4))
    print("Q1:", round(q1, 4))
    print("Median:", round(med, 4))
    print("Q3:", round(q3, 4))
    print("Maximum:", round(maximum, 4))
    print("IQR:", round(iqr, 4))
    print("Outlier limits:", round(lower_limit, 4), "to",
          round(upper_limit, 4))
    print("Number of outliers:", len(outliers))
    print("Outliers:", outliers)

    box_data.append(values)

plt.figure()
plt.boxplot(box_data, tick_labels=list(features.keys()))
plt.title("Box Plot Analysis of GHI Factors")
plt.ylabel("Attribute Value")
plt.tight_layout()
plt.show()

print("\nSPREAD COMPARISON")
for name, column in features.items():
    values = clean_values(column)
    if values:
        print(name, "Range =", round(max(values) - min(values), 4),
              "IQR =", round(percentile(values, 0.75)
                             - percentile(values, 0.25), 4))

print("Larger IQR means a wider spread in the middle 50% of values.")
