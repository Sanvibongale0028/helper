
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("flavors_of_cacao.csv")
df.columns = [str(c).strip() for c in df.columns]


def find_column(names):
    for name in names:
        for col in df.columns:
            if col.lower().strip() == name.lower().strip():
                return col
    raise ValueError("Column not found. Available columns: " + str(list(df.columns)))


rating_col = find_column(["Rating"])
cocoa_col = find_column(["Cocoa Percent", "Cocoa_Percent"])


def get_values(column):
    values = []

    for value in df[column].tolist():
        try:
            number = float(str(value).replace("%", "").strip())
            if math.isfinite(number):
                values.append(number)
            else:
                values.append(None)
        except (ValueError, TypeError):
            values.append(None)

    return values


def mean(values):
    total = 0.0
    count = 0

    for value in values:
        if value is not None:
            total += value
            count += 1

    return total / count if count else 0.0


def std_dev(values):
    avg = mean(values)
    total = 0.0
    count = 0

    for value in values:
        if value is not None:
            total += (value - avg) ** 2
            count += 1

    return math.sqrt(total / count) if count else 0.0


def minmax(values):
    valid = [v for v in values if v is not None]
    minimum = min(valid)
    maximum = max(valid)
    result = []

    for value in values:
        if value is None:
            result.append(None)
        elif maximum == minimum:
            result.append(0.0)
        else:
            result.append((value - minimum) / (maximum - minimum))

    return result


def zscore(values):
    avg = mean(values)
    sd = std_dev(values)
    result = []

    for value in values:
        if value is None:
            result.append(None)
        elif sd == 0:
            result.append(0.0)
        else:
            result.append((value - avg) / sd)

    return result


def plot_histograms(original, normalized, attribute, method):
    original_valid = [v for v in original if v is not None]
    normalized_valid = [v for v in normalized if v is not None]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(original_valid, bins=15, edgecolor="black")
    plt.title(attribute + " Before Normalization")
    plt.xlabel(attribute)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(normalized_valid, bins=15, edgecolor="black")
    plt.title(attribute + " After " + method)
    plt.xlabel("Normalized Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


for attribute, column in [
    ("Rating", rating_col),
    ("Cocoa Percent", cocoa_col)
]:
    original = get_values(column)

    print("\nATTRIBUTE:", attribute)
    print("Original min:", min(v for v in original if v is not None))
    print("Original max:", max(v for v in original if v is not None))

    minmax_values = minmax(original)
    zscore_values = zscore(original)

    print("Min-Max normalized min:",
          min(v for v in minmax_values if v is not None))
    print("Min-Max normalized max:",
          max(v for v in minmax_values if v is not None))

    print("Z-score mean:", round(mean(zscore_values), 4))
    print("Z-score standard deviation:", round(std_dev(zscore_values), 4))

    plot_histograms(original, minmax_values, attribute, "Min-Max")
    plot_histograms(original, zscore_values, attribute, "Z-score")

print("\nCONCLUSION")
print("Normalization changes the scale of the data, not its overall shape.")
print("Min-Max applies a positive linear transformation.")
print("Z-score shifts the mean to 0 and scales the standard deviation to 1.")
