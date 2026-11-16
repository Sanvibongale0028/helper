
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("flavors_of_cacao.csv")
df.columns = [str(c).strip() for c in df.columns]


def find_column(candidates):
    for candidate in candidates:
        for col in df.columns:
            if col.lower().strip() == candidate.lower().strip():
                return col
    raise ValueError(
        "Column not found: " + str(candidates)
        + "\nAvailable columns: " + str(list(df.columns))
    )


rating_col = find_column(["Rating"])
cocoa_col = find_column(["Cocoa Percent", "Cocoa_Percent"])
location_col = find_column(["Company Location"])


def clean_numeric(column, remove_percent=False):
    values = []

    for value in df[column].tolist():
        try:
            text = str(value).strip()

            if remove_percent:
                text = text.replace("%", "")

            number = float(text)

            if math.isfinite(number):
                values.append(number)

        except (ValueError, TypeError):
            pass

    return values


def median(values):
    values = sorted(values)
    n = len(values)

    if n == 0:
        return None

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

    return values[lower] + fraction * (
        values[upper] - values[lower]
    )


def five_number_summary(values):
    values = sorted(values)

    return (
        values[0],
        percentile(values, 0.25),
        median(values),
        percentile(values, 0.75),
        values[-1]
    )


def analyze_attribute(name, values):
    if not values:
        print("\nNo valid data for", name)
        return

    minimum, q1, med, q3, maximum = five_number_summary(values)

    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = [
        value for value in values
        if value < lower_limit or value > upper_limit
    ]

    print("\n", name)
    print("Minimum:", round(minimum, 4))
    print("Q1:", round(q1, 4))
    print("Median:", round(med, 4))
    print("Q3:", round(q3, 4))
    print("Maximum:", round(maximum, 4))
    print("IQR:", round(iqr, 4))
    print("Lower outlier limit:", round(lower_limit, 4))
    print("Upper outlier limit:", round(upper_limit, 4))
    print("Number of outliers:", len(outliers))
    print("Outliers:", outliers)


# Five-number summary and outliers
rating_values = clean_numeric(rating_col)
cocoa_values = clean_numeric(cocoa_col, remove_percent=True)

analyze_attribute("Rating", rating_values)
analyze_attribute("Cocoa Percentage", cocoa_values)


# Box plots of Rating and Cocoa Percentage
plt.figure(figsize=(8, 5))
plt.boxplot(
    [rating_values, cocoa_values],
    labels=["Rating", "Cocoa %"]
)
plt.title("Box Plots of Rating and Cocoa Percentage")
plt.ylabel("Value")
plt.tight_layout()
plt.show()


# Find the top five company locations by number of reviews
location_counts = {}

for value in df[location_col].tolist():
    location = str(value).strip()

    if not location or location.lower() == "nan":
        continue

    location_counts[location] = (
        location_counts.get(location, 0) + 1
    )

top_locations = sorted(
    location_counts.items(),
    key=lambda item: item[1],
    reverse=True
)[:5]

print("\nTOP FIVE COMPANY LOCATIONS BY REVIEW COUNT")

for location, count in top_locations:
    print(location, ":", count, "reviews")


# Collect ratings for the five locations
location_ratings = []
location_names = []

for location, count in top_locations:
    values = []

    for _, row in df.iterrows():
        if str(row[location_col]).strip() != location:
            continue

        try:
            rating = float(row[rating_col])

            if math.isfinite(rating):
                values.append(rating)

        except (ValueError, TypeError):
            pass

    if values:
        location_names.append(location)
        location_ratings.append(values)


# Side-by-side box plots
plt.figure(figsize=(11, 6))
plt.boxplot(location_ratings, labels=location_names)
plt.title("Rating Distribution for Top Five Company Locations")
plt.xlabel("Company Location")
plt.ylabel("Rating")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
