
import pandas as pd
import math

df = pd.read_csv("happiness.csv")
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


economy_col = find_column([
    "Economy", "Economy (GDP per Capita)",
    "Logged GDP per capita", "GDP per capita"
])

family_col = find_column(["Family", "Social support"])

health_col = find_column([
    "Health", "Health (Life Expectancy)",
    "Healthy life expectancy"
])

freedom_col = find_column([
    "Freedom", "Freedom to make life choices"
])

columns = [economy_col, family_col, health_col, freedom_col]
feature_names = ["Economy", "Family", "Health", "Freedom"]


# Select the first six records from the dataset
first_six = df.iloc[:6]

points = []
point_names = []

for index, (_, row) in enumerate(first_six.iterrows()):
    values = []

    for col in columns:
        try:
            values.append(float(row[col]))
        except (ValueError, TypeError):
            raise ValueError(
                "Missing or invalid value in the first six records. "
                "Check the dataset before running the program."
            )

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid numeric value in the first six records.")

    points.append(values)
    point_names.append("Point " + str(index + 1))

if len(points) != 6:
    raise ValueError("The dataset must contain at least six records.")


print("FIRST SIX DATA RECORDS")
print("Attributes:", feature_names)

for i in range(6):
    print(point_names[i], points[i])


# Calculate the centroid manually
centroid = []

for j in range(4):
    total = 0.0

    for i in range(6):
        total += points[i][j]

    centroid.append(total / 6)

print("\nCENTROID")

for j in range(4):
    print(feature_names[j], "=", round(centroid[j], 4))


# Distance calculations
def euclidean(a, b):
    total = 0.0

    for j in range(len(a)):
        total += (a[j] - b[j]) ** 2

    return math.sqrt(total)


def manhattan(a, b):
    total = 0.0

    for j in range(len(a)):
        difference = a[j] - b[j]

        if difference < 0:
            difference = -difference

        total += difference

    return total


def minkowski_p3(a, b):
    total = 0.0

    for j in range(len(a)):
        difference = abs(a[j] - b[j])
        total += difference ** 3

    return total ** (1 / 3)


# Calculate each point's distance from the centroid
distance_results = {
    "Euclidean": [],
    "Manhattan": [],
    "Minkowski p=3": []
}

print("\nDISTANCES FROM CENTROID")
print(
    "{:<12} {:>14} {:>14} {:>16}".format(
        "Point", "Euclidean", "Manhattan", "Minkowski p=3"
    )
)

for i in range(6):
    e = euclidean(points[i], centroid)
    m = manhattan(points[i], centroid)
    p3 = minkowski_p3(points[i], centroid)

    distance_results["Euclidean"].append(e)
    distance_results["Manhattan"].append(m)
    distance_results["Minkowski p=3"].append(p3)

    print(
        "{:<12} {:>14.4f} {:>14.4f} {:>16.4f}".format(
            point_names[i], e, m, p3
        )
    )


# Pairwise Euclidean distance matrix in lower triangular form
print("\nLOWER TRIANGULAR PAIRWISE EUCLIDEAN DISTANCE MATRIX")

print("{:>10}".format(""), end="")

for name in point_names:
    print("{:>12}".format(name), end="")

print()

for i in range(6):
    print("{:>10}".format(point_names[i]), end="")

    for j in range(6):
        if j > i:
            print("{:>12}".format(""), end="")
        else:
            distance_value = euclidean(points[i], points[j])
            print("{:>12.4f}".format(distance_value), end="")

    print()


# Nearest and farthest points from the centroid
print("\nNEAREST AND FARTHEST POINTS FROM CENTROID")

for measure, values in distance_results.items():
    nearest_index = values.index(min(values))
    farthest_index = values.index(max(values))

    print("\n", measure)
    print(
        "Nearest:",
        point_names[nearest_index],
        "Distance =", round(values[nearest_index], 4)
    )
    print(
        "Farthest:",
        point_names[farthest_index],
        "Distance =", round(values[farthest_index], 4)
    )
