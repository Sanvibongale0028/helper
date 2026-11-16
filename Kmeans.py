
import csv
import math
import matplotlib.pyplot as plt

FILE = "happiness.csv"
K = 3
MAX_ITERATIONS = 100

rows = []

with open(FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        cleaned = {
            key.strip().lower(): value.strip()
            for key, value in row.items()
            if key and value is not None
        }
        rows.append(cleaned)


def find_column(names):
    for name in names:
        if name.lower() in rows[0]:
            return name.lower()

    raise ValueError(
        "Required column not found. Available columns: "
        + str(list(rows[0].keys()))
    )


economy_col = find_column([
    "economy", "economy (gdp per capita)",
    "gdp per capita", "logged gdp per capita"
])

family_col = find_column([
    "family", "social support"
])

health_col = find_column([
    "health", "health (life expectancy)",
    "healthy life expectancy"
])

freedom_col = find_column([
    "freedom", "freedom to make life choices"
])

score_col = find_column([
    "h_score", "happiness score",
    "happiness.score", "score", "ladder score"
])

features = [
    economy_col, family_col, health_col, freedom_col
]

countries = []
X = []
happiness_scores = []

for row in rows:
    try:
        values = [float(row[c]) for c in features]
        score = float(row[score_col])

        if all(math.isfinite(v) for v in values) and math.isfinite(score):
            X.append(values)
            happiness_scores.append(score)
            countries.append(row.get("country", "Unknown"))

    except (ValueError, KeyError):
        continue

if len(X) < K:
    raise ValueError("At least three valid records are required.")

print("Total countries:", len(X))


# Standardize features manually
n = len(X)
p = len(features)

means = []
stds = []

for j in range(p):
    column = [X[i][j] for i in range(n)]
    mean = sum(column) / n

    variance = sum(
        (value - mean) ** 2 for value in column
    ) / n

    sd = math.sqrt(variance)

    if sd == 0:
        sd = 1.0

    means.append(mean)
    stds.append(sd)

Z = []

for row in X:
    Z.append([
        (row[j] - means[j]) / stds[j]
        for j in range(p)
    ])


# Euclidean distance
def distance(a, b):
    total = 0.0

    for j in range(len(a)):
        total += (a[j] - b[j]) ** 2

    return math.sqrt(total)


# Initialize centres using farthest-point selection
centres = [Z[0][:]]

while len(centres) < K:
    best_point = None
    best_distance = -1

    for point in Z:
        nearest = min(
            distance(point, centre) for centre in centres
        )

        if nearest > best_distance:
            best_distance = nearest
            best_point = point

    centres.append(best_point[:])


# K-Means algorithm
labels = [-1] * n

for iteration in range(MAX_ITERATIONS):
    new_labels = []

    # Assign points to the nearest centre
    for point in Z:
        distances = [
            distance(point, centre) for centre in centres
        ]

        new_labels.append(distances.index(min(distances)))

    if new_labels == labels:
        break

    labels = new_labels

    # Recalculate centres
    new_centres = []

    for cluster in range(K):
        members = [
            Z[i] for i in range(n)
            if labels[i] == cluster
        ]

        if not members:
            new_centres.append(centres[cluster][:])
            continue

        centre = []

        for j in range(p):
            centre.append(
                sum(row[j] for row in members) / len(members)
            )

        new_centres.append(centre)

    centres = new_centres


print("\nNUMBER OF ITERATIONS:", iteration + 1)


# Print cluster centres in original feature units
print("\nCLUSTER CENTRES")

for cluster in range(K):
    print("Cluster", cluster + 1)

    for j in range(p):
        original_value = centres[cluster][j] * stds[j] + means[j]

        print(features[j], ":", round(original_value, 4))


# Print cluster sizes and mean happiness scores
print("\nCLUSTER SIZES AND MEAN HAPPINESS SCORES")

for cluster in range(K):
    members = [
        i for i in range(n) if labels[i] == cluster
    ]

    if members:
        mean_score = sum(
            happiness_scores[i] for i in members
        ) / len(members)
    else:
        mean_score = 0.0

    print("Cluster:", cluster + 1)
    print("Size:", len(members))
    print("Mean H_Score:", round(mean_score, 4))


# Plot Economy versus Health
plt.figure()

for cluster in range(K):
    economy_values = [
        X[i][0] for i in range(n) if labels[i] == cluster
    ]

    health_values = [
        X[i][2] for i in range(n) if labels[i] == cluster
    ]

    plt.scatter(
        economy_values,
        health_values,
        label="Cluster " + str(cluster + 1)
    )

plt.xlabel("Economy")
plt.ylabel("Health")
plt.title("K-Means Clusters: Economy vs Health")
plt.legend()
plt.tight_layout()
plt.show()


# Calculate Silhouette Coefficient manually
# a(i): mean distance to other points in the same cluster
# b(i): smallest mean distance to another cluster

silhouette_values = []

for i in range(n):
    own_cluster = labels[i]

    same_cluster_distances = [
        distance(Z[i], Z[j])
        for j in range(n)
        if labels[j] == own_cluster and j != i
    ]

    if not same_cluster_distances:
        silhouette_values.append(0.0)
        continue

    a = sum(same_cluster_distances) / len(
        same_cluster_distances
    )

    b = float("inf")

    for other_cluster in range(K):
        if other_cluster == own_cluster:
            continue

        other_distances = [
            distance(Z[i], Z[j])
            for j in range(n)
            if labels[j] == other_cluster
        ]

        if other_distances:
            average_distance = sum(other_distances) / len(
                other_distances
            )

            b = min(b, average_distance)

    if b == float("inf"):
        silhouette_values.append(0.0)
    elif max(a, b) == 0:
        silhouette_values.append(0.0)
    else:
        silhouette_values.append(
            (b - a) / max(a, b)
        )


silhouette_score = sum(silhouette_values) / n

print("\nSILHOUETTE COEFFICIENT:")
print(round(silhouette_score, 4))

if silhouette_score > 0.7:
    print("Clusters are well separated.")
elif silhouette_score > 0.5:
    print("Clusters are reasonably separated.")
elif silhouette_score > 0.25:
    print("Clusters have weak to moderate separation.")
else:
    print("Clusters may overlap or be poorly separated.")
