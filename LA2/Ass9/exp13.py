
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load dataset
df = pd.read_csv("happiness.csv")
df.columns = [str(c).strip() for c in df.columns]


# Find columns using common dataset names
def find_column(names):
    for name in names:
        for col in df.columns:
            if col.lower().strip() == name.lower().strip():
                return col
    raise ValueError(
        "Column not found: " + str(names)
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


# Extract complete numeric records
X_original = []

for _, row in df.iterrows():
    values = []

    try:
        for col in columns:
            value = float(row[col])

            if not math.isfinite(value):
                raise ValueError("Invalid number")

            values.append(value)

        X_original.append(values)

    except (ValueError, TypeError):
        continue

if len(X_original) < 6:
    raise ValueError("At least 6 valid records are required.")

print("Number of records:", len(X_original))
print("Features:", feature_names)


# Z-score normalization implemented manually
def mean(values):
    total = 0.0

    for value in values:
        total += value

    return total / len(values)


def standard_deviation(values):
    avg = mean(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return math.sqrt(total / len(values))


X = []
feature_means = []
feature_stds = []

for j in range(4):
    values = [row[j] for row in X_original]
    feature_means.append(mean(values))
    feature_stds.append(standard_deviation(values))

for row in X_original:
    normalized_row = []

    for j in range(4):
        if feature_stds[j] == 0:
            normalized_row.append(0.0)
        else:
            normalized_row.append(
                (row[j] - feature_means[j]) / feature_stds[j]
            )

    X.append(normalized_row)


# Euclidean distance
def distance(a, b):
    total = 0.0

    for j in range(len(a)):
        total += (a[j] - b[j]) ** 2

    return math.sqrt(total)


# K-Means implementation
def kmeans(data, k, max_iterations=100):
    n = len(data)

    # Deterministic farthest-point initialization
    centres = [data[0][:]]

    while len(centres) < k:
        farthest_point = None
        farthest_distance = -1.0

        for point in data:
            nearest_distance = min(
                distance(point, centre) for centre in centres
            )

            if nearest_distance > farthest_distance:
                farthest_distance = nearest_distance
                farthest_point = point

        centres.append(farthest_point[:])

    labels = [-1] * n

    for iteration in range(max_iterations):
        new_labels = []

        # Assign each point to its nearest centre
        for point in data:
            distances = [
                distance(point, centre) for centre in centres
            ]

            new_labels.append(distances.index(min(distances)))

        if new_labels == labels:
            break

        labels = new_labels
        new_centres = []

        # Update cluster centres
        for cluster in range(k):
            members = [
                data[i] for i in range(n)
                if labels[i] == cluster
            ]

            if not members:
                new_centres.append(centres[cluster][:])
                continue

            centre = []

            for j in range(len(data[0])):
                total = 0.0

                for member in members:
                    total += member[j]

                centre.append(total / len(members))

            new_centres.append(centre)

        centres = new_centres

    return labels, centres


# Calculate SSE
def calculate_sse(data, labels, centres):
    sse = 0.0

    for i in range(len(data)):
        cluster = labels[i]

        for j in range(len(data[i])):
            difference = data[i][j] - centres[cluster][j]
            sse += difference * difference

    return sse


# Calculate Silhouette Coefficient manually
def silhouette_score(data, labels, k):
    all_scores = []

    for i in range(len(data)):
        own_cluster = labels[i]

        same_cluster_distances = []

        for j in range(len(data)):
            if i != j and labels[j] == own_cluster:
                same_cluster_distances.append(
                    distance(data[i], data[j])
                )

        # Silhouette is defined as zero for a singleton cluster
        if not same_cluster_distances:
            all_scores.append(0.0)
            continue

        a = sum(same_cluster_distances) / len(
            same_cluster_distances
        )

        b = float("inf")

        for cluster in range(k):
            if cluster == own_cluster:
                continue

            other_distances = []

            for j in range(len(data)):
                if labels[j] == cluster:
                    other_distances.append(
                        distance(data[i], data[j])
                    )

            if other_distances:
                avg_distance = sum(other_distances) / len(
                    other_distances
                )

                if avg_distance < b:
                    b = avg_distance

        if b == float("inf") or max(a, b) == 0:
            all_scores.append(0.0)
        else:
            all_scores.append((b - a) / max(a, b))

    return sum(all_scores) / len(all_scores)


# Test k from 2 to 6
sse_values = []
silhouette_values = []
results = {}

print("\nK-MEANS EVALUATION")

for k in range(2, 7):
    labels, centres = kmeans(X, k)

    sse = calculate_sse(X, labels, centres)
    silhouette = silhouette_score(X, labels, k)

    sse_values.append(sse)
    silhouette_values.append(silhouette)
    results[k] = (labels, centres)

    print(
        "K =", k,
        "| SSE =", round(sse, 4),
        "| Silhouette =", round(silhouette, 4)
    )


# Plot elbow curve
k_values = [2, 3, 4, 5, 6]

plt.figure()
plt.plot(k_values, sse_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot silhouette scores
plt.figure()
plt.plot(k_values, silhouette_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Coefficient")
plt.title("Silhouette Score vs K")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()


# Select k with the highest silhouette score
best_index = silhouette_values.index(max(silhouette_values))
best_k = k_values[best_index]

final_labels, final_centres = results[best_k]

print("\nSelected K:", best_k)
print("Reason: It has the highest silhouette score among k = 2 to 6.")
print("Also inspect the elbow curve for the point where SSE")
print("starts decreasing more slowly.")


# Plot Economy vs Health using the final clusters
plt.figure()

for cluster in range(best_k):
    economy_values = []
    health_values = []

    for i in range(len(X_original)):
        if final_labels[i] == cluster:
            economy_values.append(X_original[i][0])
            health_values.append(X_original[i][2])

    plt.scatter(
        economy_values,
        health_values,
        label="Cluster " + str(cluster + 1)
    )

plt.xlabel("Economy")
plt.ylabel("Health")
plt.title("Economy vs Health - K-Means Clusters")
plt.legend()
plt.tight_layout()
plt.show()


# Describe each cluster using original feature values
print("\nFINAL CLUSTER DESCRIPTIONS")

for cluster in range(best_k):
    members = [
        X_original[i] for i in range(len(X_original))
        if final_labels[i] == cluster
    ]

    print("\nCluster", cluster + 1)
    print("Number of countries:", len(members))

    if not members:
        print("Empty cluster")
        continue

    for j in range(4):
        values = [member[j] for member in members]
        print(
            feature_names[j] + " mean:",
            round(mean(values), 4)
        )
