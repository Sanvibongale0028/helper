
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

df = pd.read_csv("happiness.csv")
df.columns = df.columns.str.strip()

def find_column(names):
    for name in names:
        for col in df.columns:
            if col.lower().strip() == name.lower():
                return col
    for name in names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    raise ValueError("Column not found: " + str(names))

economy_col = find_column(["Economy", "Economy (GDP per Capita)", "GDP per capita"])
family_col = find_column(["Fam", "Family", "Social support"])
health_col = find_column(["Health", "Health (Life Expectancy)", "Healthy life expectancy"])
freedom_col = find_column(["Freedom", "Freedom to make life choices"])
score_col = find_column(["H_Score", "H Score", "Happiness Score", "Score"])

features = [economy_col, family_col, health_col, freedom_col]

for col in features + [score_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features + [score_col]).reset_index(drop=True)

# Standardise features manually using population standard deviation
data = []
means = []
stds = []

for col in features:
    values = df[col].tolist()
    mean_value = sum(values) / len(values)
    std_value = math.sqrt(
        sum((x - mean_value) ** 2 for x in values) / len(values)
    )

    means.append(mean_value)
    stds.append(std_value)

    if std_value == 0:
        data.append([0.0] * len(values))
    else:
        data.append([(x - mean_value) / std_value for x in values])

points = [
    [data[j][i] for j in range(len(features))]
    for i in range(len(df))
]

def euclidean(a, b):
    return math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(len(a))))

random.seed(42)
k = 3
centres = random.sample(points, k)

# K-Means
for iteration in range(100):
    clusters = [[] for _ in range(k)]
    labels = []

    for point in points:
        distances = [euclidean(point, centre) for centre in centres]
        cluster_id = distances.index(min(distances))
        clusters[cluster_id].append(point)
        labels.append(cluster_id)

    new_centres = []

    for cluster in clusters:
        if not cluster:
            new_centres.append(random.choice(points))
            continue

        centre = [
            sum(point[j] for point in cluster) / len(cluster)
            for j in range(len(features))
        ]
        new_centres.append(centre)

    if all(
        euclidean(centres[i], new_centres[i]) < 1e-8
        for i in range(k)
    ):
        centres = new_centres
        break

    centres = new_centres

# Reassign points to final centres
labels = []
clusters = [[] for _ in range(k)]

for point in points:
    distances = [euclidean(point, centre) for centre in centres]
    cluster_id = distances.index(min(distances))
    labels.append(cluster_id)
    clusters[cluster_id].append(point)

df["Cluster"] = labels

print("CLUSTER CENTRES (original feature scale)")
for i, centre in enumerate(centres):
    original_centre = [
        centre[j] * stds[j] + means[j]
        for j in range(len(features))
    ]
    print("Cluster", i + 1, dict(zip(features, [
        round(x, 4) for x in original_centre
    ])))

print("\nCLUSTER SIZES")
for i in range(k):
    print("Cluster", i + 1, ":", labels.count(i))

print("\nMEAN H_Score PER CLUSTER")
for i in range(k):
    scores = df.loc[df["Cluster"] == i, score_col].tolist()
    if scores:
        print("Cluster", i + 1, ":", round(sum(scores) / len(scores), 4))
    else:
        print("Cluster", i + 1, ": empty")

# Silhouette coefficient
def silhouette_for_point(index):
    own_cluster = labels[index]
    own_points = [
        j for j in range(len(points))
        if labels[j] == own_cluster and j != index
    ]

    if not own_points:
        return 0.0

    a = sum(
        euclidean(points[index], points[j])
        for j in own_points
    ) / len(own_points)

    other_averages = []

    for cluster_id in range(k):
        if cluster_id == own_cluster:
            continue

        other_points = [
            j for j in range(len(points))
            if labels[j] == cluster_id
        ]

        if other_points:
            avg_distance = sum(
                euclidean(points[index], points[j])
                for j in other_points
            ) / len(other_points)

            other_averages.append(avg_distance)

    if not other_averages:
        return 0.0

    b = min(other_averages)

    if max(a, b) == 0:
        return 0.0

    return (b - a) / max(a, b)

silhouettes = [
    silhouette_for_point(i)
    for i in range(len(points))
]

print("\nSilhouette Coefficient:", round(
    sum(silhouettes) / len(silhouettes), 4
))

# Plot Economy vs Health coloured by cluster
for cluster_id in range(k):
    subset = df[df["Cluster"] == cluster_id]

    plt.scatter(
        subset[economy_col],
        subset[health_col],
        label="Cluster " + str(cluster_id + 1)
    )

plt.xlabel("Economy")
plt.ylabel("Health")
plt.title("K-Means Clustering (k=3)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
