
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# ============================================================
# PART A: COCOA DATA VISUALIZATION
# ============================================================

df = pd.read_csv("flavors_of_cacao.csv")
df.columns = [str(c).strip() for c in df.columns]

print("PART A: DATASET CHARACTERISTICS")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", list(df.columns))
print("\nMissing values:")
print(df.isnull().sum())


def find_column(columns, candidates):
    for candidate in candidates:
        for column in columns:
            if column.lower().strip() == candidate.lower().strip():
                return column

    raise ValueError(
        "Column not found: " + str(candidates)
        + "\nAvailable columns: " + str(list(columns))
    )


company_col = find_column(
    df.columns, ["Company (Maker-if known)", "Company"]
)
location_col = find_column(
    df.columns, ["Company Location"]
)
year_col = find_column(
    df.columns, ["Review Date"]
)
cocoa_col = find_column(
    df.columns, ["Cocoa Percent"]
)
rating_col = find_column(
    df.columns, ["Rating"]
)

print("\nATTRIBUTE TYPES")
print("Company: Nominal")
print("Company Location: Nominal")
print("Review Date: Interval")
print("Cocoa Percent: Ratio")
print("Rating: Ordinal")
print("Bean Type: Nominal")
print("Broad Bean Origin: Nominal")


# Prepare valid records
records = []

for _, row in df.iterrows():
    try:
        year = int(float(row[year_col]))
        rating = float(row[rating_col])
        cocoa = float(
            str(row[cocoa_col]).replace("%", "").strip()
        )

        if (
            math.isfinite(rating)
            and math.isfinite(cocoa)
            and 0 <= rating <= 5
            and 0 <= cocoa <= 100
        ):
            records.append({
                "year": year,
                "rating": rating,
                "cocoa": cocoa,
                "location": str(row[location_col]).strip()
            })

    except (ValueError, TypeError):
        continue


# 1. Line chart: number of reviews per year
year_counts = {}

for record in records:
    year = record["year"]
    year_counts[year] = year_counts.get(year, 0) + 1

sorted_years = sorted(year_counts)

plt.figure()
plt.plot(
    sorted_years,
    [year_counts[y] for y in sorted_years],
    marker="o"
)
plt.xlabel("Review Year")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews per Year")
plt.tight_layout()
plt.show()

print("\nReviews per year:")
for year in sorted_years:
    print(year, year_counts[year])


# 2. Bar chart: top 10 locations with at least 20 reviews
location_stats = {}

for record in records:
    location = record["location"]

    if not location or location.lower() == "nan":
        continue

    if location not in location_stats:
        location_stats[location] = [0.0, 0]

    location_stats[location][0] += record["rating"]
    location_stats[location][1] += 1

eligible_locations = []

for location, values in location_stats.items():
    rating_sum, count = values

    if count >= 20:
        mean_rating = rating_sum / count
        eligible_locations.append(
            (location, mean_rating, count)
        )

eligible_locations.sort(
    key=lambda item: item[1],
    reverse=True
)

top_locations = eligible_locations[:10]

print("\nTOP 10 HIGHEST-RATED LOCATIONS")
for location, mean_rating, count in top_locations:
    print(
        location,
        "Mean rating:", round(mean_rating, 3),
        "Reviews:", count
    )

plt.figure()
plt.bar(
    [x[0] for x in top_locations],
    [x[1] for x in top_locations]
)
plt.xlabel("Company Location")
plt.ylabel("Mean Rating")
plt.title("Top 10 Highest-Rated Locations (Minimum 20 Reviews)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 3. Pie chart: rating classes
# Low: below 2.5, Medium: 2.5 to below 3.5, High: 3.5 and above
rating_classes = {
    "Low (<2.5)": 0,
    "Medium (2.5-3.49)": 0,
    "High (>=3.5)": 0
}

for record in records:
    rating = record["rating"]

    if rating < 2.5:
        rating_classes["Low (<2.5)"] += 1
    elif rating < 3.5:
        rating_classes["Medium (2.5-3.49)"] += 1
    else:
        rating_classes["High (>=3.5)"] += 1

print("\nRATING CLASSES:", rating_classes)

plt.figure()
plt.pie(
    list(rating_classes.values()),
    labels=list(rating_classes.keys()),
    autopct="%1.1f%%"
)
plt.title("Distribution of Rating Classes")
plt.tight_layout()
plt.show()


# 4. Histogram: Cocoa percentage
cocoa_values = [record["cocoa"] for record in records]

plt.figure()
plt.hist(cocoa_values, bins=10, edgecolor="black")
plt.xlabel("Cocoa Percentage")
plt.ylabel("Frequency")
plt.title("Distribution of Cocoa Percentage")
plt.tight_layout()
plt.show()


# 5. Scatter plot: Review Date versus mean Rating
year_rating_sum = {}
year_rating_count = {}

for record in records:
    year = record["year"]

    year_rating_sum[year] = (
        year_rating_sum.get(year, 0) + record["rating"]
    )

    year_rating_count[year] = (
        year_rating_count.get(year, 0) + 1
    )

mean_rating_by_year = {
    year: year_rating_sum[year] / year_rating_count[year]
    for year in year_rating_sum
}

plt.figure()
plt.scatter(
    list(mean_rating_by_year.keys()),
    list(mean_rating_by_year.values())
)
plt.xlabel("Review Date (Year)")
plt.ylabel("Mean Rating")
plt.title("Review Date versus Mean Rating")
plt.tight_layout()
plt.show()

