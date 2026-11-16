
import csv
import math

# ------------------------------------------
# PART B: CENTROID AND DISTANCE MEASURES
# ------------------------------------------

filename = "2019.csv"

attributes = [
    "H_Score",
    "GDP",
    "Social_Support",
    "Life_Expectancy",
    "Freedom"
]

data = []

# Read the dataset
with open(filename, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        record = [
            float(row["Score"]),
            float(row["GDP per capita"]),
            float(row["Social support"]),
            float(row["Healthy life expectancy"]),
            float(row["Freedom to make life choices"])
        ]

        data.append(record)

# Select rows 51 to 56 (1-based indexing)
selected = data[50:56]

if len(selected) != 6:
    raise ValueError("Dataset must contain at least 56 rows.")

print("SELECTED ROWS (51 TO 56)")
for i in range(len(selected)):
    print("Row", 51 + i, ":", selected[i])

# 1. Calculate centroid
centroid = []

for j in range(len(attributes)):
    total = 0

    for i in range(len(selected)):
        total += selected[i][j]

    centroid.append(total / len(selected))

print("\nCENTROID")
for j in range(len(attributes)):
    print(attributes[j], ":", centroid[j])

# 2. Distance functions implemented manually
def euclidean_distance(point, center):
    total = 0

    for j in range(len(point)):
        difference = point[j] - center[j]
        total += difference * difference

    return math.sqrt(total)


def manhattan_distance(point, center):
    total = 0

    for j in range(len(point)):
        total += abs(point[j] - center[j])

    return total


def minkowski_distance(point, center, p):
    total = 0

    for j in range(len(point)):
        difference = abs(point[j] - center[j])
        total += difference ** p

    return total ** (1 / p)


# 3. Calculate distances of each row from centroid
euclidean = []
manhattan = []
minkowski = []

print("\nDISTANCES FROM CENTROID")
print("Row\tEuclidean\tManhattan\tMinkowski(p=3)")

for i in range(len(selected)):
    point = selected[i]

    e = euclidean_distance(point, centroid)
    m = manhattan_distance(point, centroid)
    k = minkowski_distance(point, centroid, 3)

    euclidean.append(e)
    manhattan.append(m)
    minkowski.append(k)

    print(
        51 + i,
        "\t", round(e, 4),
        "\t\t", round(m, 4),
        "\t\t", round(k, 4)
    )

# 4. Lower triangular Euclidean distance matrix
n = len(selected)

print("\nLOWER TRIANGULAR EUCLIDEAN DISTANCE MATRIX")
print("Rows and columns correspond to dataset rows 51-56")

for i in range(n):
    for j in range(n):
        if j <= i:
            distance = euclidean_distance(
                selected[i], selected[j]
            )
            print("{:10.4f}".format(distance), end=" ")
        else:
            print(" " * 11, end="")

    print()

# 5. Find nearest and farthest rows from centroid
def find_nearest_and_farthest(distances):
    nearest_index = 0
    farthest_index = 0

    for i in range(1, len(distances)):
        if distances[i] < distances[nearest_index]:
            nearest_index = i

        if distances[i] > distances[farthest_index]:
            farthest_index = i

    return nearest_index, farthest_index


distance_sets = [
    ("Euclidean", euclidean),
    ("Manhattan", manhattan),
    ("Minkowski", minkowski)
]

nearest_rows = []
farthest_rows = []

print("\nNEAREST AND FARTHEST ROWS FROM CENTROID")

for name, distances in distance_sets:
    nearest, farthest = find_nearest_and_farthest(distances)

    nearest_row = 51 + nearest
    farthest_row = 51 + farthest

    nearest_rows.append(nearest_row)
    farthest_rows.append(farthest_row)

    print(name)
    print(
        "Nearest row:", nearest_row,
        "Distance:", round(distances[nearest], 4)
    )
    print(
        "Farthest row:", farthest_row,
        "Distance:", round(distances[farthest], 4)
    )

# 6. Check whether all measures agree
print("\nCOMPARISON OF DISTANCE MEASURES")

if nearest_rows[0] == nearest_rows[1] == nearest_rows[2]:
    print("All three measures agree on the nearest row.")
else:
    print("The measures do not all agree on the nearest row.")

if farthest_rows[0] == farthest_rows[1] == farthest_rows[2]:
    print("All three measures agree on the farthest row.")
else:
    print("The measures do not all agree on the farthest row.")
