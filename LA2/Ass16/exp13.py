
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
    raise ValueError("Column not found: " + str(candidates))


company_col = find_column(["Company (Maker-if known)", "Company"])
rating_col = find_column(["Rating"])
cocoa_col = find_column(["Cocoa Percent"])


# Aggregate data at company level
company_data = {}

for _, row in df.iterrows():
    company = str(row[company_col]).strip()

    try:
        rating = float(row[rating_col])
        cocoa = float(str(row[cocoa_col]).replace("%", "").strip())

        if (not company or company.lower() == "nan"
                or not math.isfinite(rating)
                or not math.isfinite(cocoa)):
            continue
    except (ValueError, TypeError):
        continue

    if company not in company_data:
        company_data[company] = [0.0, 0.0, 0]

    company_data[company][0] += rating
    company_data[company][1] += cocoa
    company_data[company][2] += 1


companies = []
X = []

for company, values in company_data.items():
    rating_sum, cocoa_sum, count = values

    if count >= 5:
        companies.append(company)
        X.append([
            rating_sum / count,
            cocoa_sum / count,
            count
        ])

if len(X) < 3:
    raise ValueError("At least 3 companies with 5 reviews are required.")

print("Companies with at least 5 reviews:", len(X))
print("Features: Mean Rating, Mean Cocoa %, Number of Reviews")


# Min-max normalization
mins = []
maxs = []

for j in range(3):
    column_values = [row[j] for row in X]
    mins.append(min(column_values))
    maxs.append(max(column_values))

Z = []

for row in X:
    normalized = []
    for j in range(3):
        if maxs[j] == mins[j]:
            normalized.append(0.0)
        else:
            normalized.append(
                (row[j] - mins[j]) / (maxs[j] - mins[j])
            )
    Z.append(normalized)


def distance(a, b):
    total = 0.0
    for j in range(len(a)):
        total += (a[j] - b[j]) ** 2
    return math.sqrt(total)


# K-Means, k = 3
K = 3

# Farthest-point initialization
centres = [Z[0][:]]

while len(centres) < K:
    farthest = None
    farthest_dist = -1

    for point in Z:
        nearest = min(distance(point, centre) for centre in centres)

        if nearest > farthest_dist:
            farthest_dist = nearest
            farthest = point

    centres.append(farthest[:])


labels = [-1] * len(Z)

for iteration in range(100):
    new_labels = []

    for point in Z:
        distances = [distance(point, centre) for centre in centres]
        new_labels.append(distances.index(min(distances)))

    if new_labels == labels:
        break

    labels = new_labels
    new_centres = []

    for cluster in range(K):
        members = [
            Z[i] for i in range(len(Z))
            if labels[i] == cluster
        ]

        if not members:
            new_centres.append(centres[cluster][:])
        else:
            centre = []
            for j in range(3):
                total = 0.0
                for member in members:
                    total += member[j]
                centre.append(total / len(members))
            new_centres.append(centre)

    centres = new_centres

print("\nIterations:", iteration + 1)


# Cluster descriptions in original units
print("\nCLUSTER DESCRIPTIONS")

for cluster in range(K):
    members = [i for i in range(len(X)) if labels[i] == cluster]

    print("\nCluster", cluster + 1)
    print("Number of companies:", len(members))

    if not members:
        print("Empty cluster")
        continue

    mean_rating = sum(X[i][0] for i in members) / len(members)
    mean_cocoa = sum(X[i][1] for i in members) / len(members)
    mean_reviews = sum(X[i][2] for i in members) / len(members)

    print("Mean Rating:", round(mean_rating, 3))
    print("Mean Cocoa %:", round(mean_cocoa, 3))
    print("Mean Reviews:", round(mean_reviews, 3))

    if mean_rating >= 3.5:
        print("Description: Higher-rated companies.")
    elif mean_rating >= 3.0:
        print("Description: Moderately rated companies.")
    else:
        print("Description: Lower-rated companies.")


# Plot Cocoa % vs Rating
plt.figure()

for cluster in range(K):
    cocoa_values = [
        X[i][1] for i in range(len(X)) if labels[i] == cluster
    ]
    rating_values = [
        X[i][0] for i in range(len(X)) if labels[i] == cluster
    ]

    plt.scatter(
        cocoa_values, rating_values,
        label="Cluster " + str(cluster + 1)
    )

plt.xlabel("Mean Cocoa Percentage")
plt.ylabel("Mean Rating")
plt.title("K-Means Clustering of Cocoa Companies")
plt.legend()
plt.tight_layout()
plt.show()


# Silhouette Coefficient
silhouette_values = []

for i in range(len(Z)):
    own_cluster = labels[i]

    same_distances = [
        distance(Z[i], Z[j])
        for j in range(len(Z))
        if labels[j] == own_cluster and i != j
    ]

    if not same_distances:
        silhouette_values.append(0.0)
        continue

    a = sum(same_distances) / len(same_distances)
    b = float("inf")

    for cluster in range(K):
        if cluster == own_cluster:
            continue

        other_distances = [
            distance(Z[i], Z[j])
            for j in range(len(Z))
            if labels[j] == cluster
        ]

        if other_distances:
            avg_distance = sum(other_distances) / len(other_distances)
            if avg_distance < b:
                b = avg_distance

    if b == float("inf") or max(a, b) == 0:
        silhouette_values.append(0.0)
    else:
        silhouette_values.append((b - a) / max(a, b))

silhouette = sum(silhouette_values) / len(silhouette_values)

print("\nSilhouette Coefficient:", round(silhouette, 4))

if silhouette > 0.7:
    print("Clusters are well separated.")
elif silhouette > 0.5:
    print("Clusters have reasonably good separation.")
elif silhouette > 0.25:
    print("Clusters have weak to moderate separation.")
else:
    print("Clusters may overlap or be poorly separated.")
