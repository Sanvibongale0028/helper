
import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cacao.csv")

df.columns = df.columns.str.strip()

rating_col = next(
    (c for c in df.columns if c.lower() == "rating"),
    None
)

if rating_col is None:
    raise ValueError("Rating column not found.")

ratings = []
for value in df[rating_col]:
    try:
        x = float(str(value).replace(",", "").strip())
        if not math.isnan(x):
            ratings.append(x)
    except (ValueError, TypeError):
        pass

if len(ratings) == 0:
    raise ValueError("No valid ratings found.")

def median(values):
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2

# Equal-width binning
minimum = min(ratings)
maximum = max(ratings)
width = (maximum - minimum) / 3

equal_width_bins = [[], [], []]

for x in ratings:
    if width == 0:
        index = 0
    else:
        index = int((x - minimum) / width)
        if index > 2:
            index = 2
    equal_width_bins[index].append(x)

# Equal-frequency binning
sorted_ratings = sorted(ratings)
n = len(sorted_ratings)
equal_frequency_bins = [
    sorted_ratings[i * n // 3:(i + 1) * n // 3]
    for i in range(3)
]

# Smoothing functions
def smooth_by_mean(bin_values):
    mean_value = sum(bin_values) / len(bin_values)
    return [mean_value] * len(bin_values)

def smooth_by_median(bin_values):
    median_value = median(bin_values)
    return [median_value] * len(bin_values)

def smooth_by_boundaries(bin_values):
    low = min(bin_values)
    high = max(bin_values)
    result = []

    for x in bin_values:
        if abs(x - low) <= abs(x - high):
            result.append(low)
        else:
            result.append(high)

    return result

def show_results(name, bins):
    print("\n", name)
    print("-" * 65)

    for i, values in enumerate(bins):
        if len(values) == 0:
            print("Bin", i + 1, "is empty")
            continue

        print("\nBin", i + 1)
        print("Count:", len(values))
        print("Range:", min(values), "to", max(values))
        print("Original values:", values)
        print("Smoothed by mean:", smooth_by_mean(values))
        print("Smoothed by median:", smooth_by_median(values))
        print("Smoothed by boundaries:", smooth_by_boundaries(values))

show_results("Equal-Width Binning", equal_width_bins)
show_results("Equal-Frequency Binning", equal_frequency_bins)

print("\nBIN COUNT COMPARISON")
print("Equal-width counts:",
      [len(b) for b in equal_width_bins])
print("Equal-frequency counts:",
      [len(b) for b in equal_frequency_bins])

plt.figure(figsize=(8, 5))
plt.boxplot(
    [ratings,
     [x for b in equal_width_bins for x in b],
     [x for b in equal_frequency_bins for x in b]],
    tick_labels=["Original", "Equal Width", "Equal Frequency"]
)
plt.title("Rating Distribution and Binning Comparison")
plt.ylabel("Rating")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
