
import pandas as pd
import math

df = pd.read_csv("flavors_of_cacao.csv")
df.columns = df.columns.str.strip()

cocoa_col = next(
    (c for c in df.columns if c.lower() == "cocoa percent"),
    None
)

if cocoa_col is None:
    raise ValueError("Cocoa Percent column not found.")

df["Cocoa Numeric"] = df[cocoa_col].apply(
    lambda x: float(str(x).replace("%", "").strip())
    if pd.notna(x) else float("nan")
)

df = df.dropna(subset=["Cocoa Numeric"]).copy()
df = df.sort_values("Cocoa Numeric").reset_index(drop=True)

values = df["Cocoa Numeric"].tolist()
minimum = min(values)
maximum = max(values)
bin_width = (maximum - minimum) / 4

bins = [[], [], [], []]
bin_records = [[], [], [], []]

# Assign each sorted value to an equal-width bin
for index, value in enumerate(values):
    if bin_width == 0:
        bin_index = 0
    else:
        bin_index = int((value - minimum) / bin_width)
        if bin_index > 3:
            bin_index = 3

    bins[bin_index].append(value)
    bin_records[bin_index].append(index)

# Calculate median manually
def median(data):
    data = sorted(data)
    n = len(data)

    if n % 2 == 1:
        return data[n // 2]

    return (data[n // 2 - 1] + data[n // 2]) / 2

# Store smoothed values for each sorted record
mean_smoothed = [None] * len(values)
median_smoothed = [None] * len(values)
boundary_smoothed = [None] * len(values)

print("Minimum Cocoa Percent:", minimum)
print("Maximum Cocoa Percent:", maximum)
print("Equal bin width:", round(bin_width, 3))

print("\nBIN RANGES AND COUNTS")

for b in range(4):
    low = minimum + b * bin_width
    high = minimum + (b + 1) * bin_width

    if b == 3:
        range_text = f"[{low:.2f}, {high:.2f}]"
    else:
        range_text = f"[{low:.2f}, {high:.2f})"

    print(
        "Bin", b + 1,
        "| Range:", range_text,
        "| Count:", len(bins[b])
    )

    if len(bins[b]) == 0:
        continue

    bin_mean = sum(bins[b]) / len(bins[b])
    bin_median = median(bins[b])
    bin_min = min(bins[b])
    bin_max = max(bins[b])

    for record_index in bin_records[b]:
        original = values[record_index]

        mean_smoothed[record_index] = bin_mean
        median_smoothed[record_index] = bin_median

        if abs(original - bin_min) <= abs(original - bin_max):
            boundary_smoothed[record_index] = bin_min
        else:
            boundary_smoothed[record_index] = bin_max

print("\nFIRST 15 SORTED RECORDS WITH SMOOTHED VALUES")

sample = pd.DataFrame({
    "Original Cocoa %": values[:15],
    "Bin Mean": mean_smoothed[:15],
    "Bin Median": median_smoothed[:15],
    "Bin Boundary": boundary_smoothed[:15]
})

print(sample.round(2).to_string(index=False))
