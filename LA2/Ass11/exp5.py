
import csv
import math

FILE = "World-happiness-report-2024.csv"

features = [
    "Log GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices"
]

# Read the CSV file
data = []

with open(FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        record = {"Country": row.get("Country name", "Unknown")}

        for feature in features:
            value = row.get(feature, "").strip()

            try:
                record[feature] = float(value) if value else None
            except ValueError:
                record[feature] = None

        data.append(record)

if not data:
    raise ValueError("The CSV file is empty.")

# Calculate median manually
def median(values):
    values = sorted(values)
    n = len(values)

    if n == 0:
        return 0.0

    if n % 2 == 1:
        return values[n // 2]

    return (values[n // 2 - 1] + values[n // 2]) / 2


# Calculate percentile using linear interpolation
def percentile(values, percentage):
    values = sorted(values)
    position = (len(values) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] + fraction * (
        values[upper] - values[lower]
    )


# Step 1: Replace missing values with column median
print("MISSING VALUES BEFORE IMPUTATION")

for feature in features:
    missing_count = sum(
        1 for row in data if row[feature] is None
    )

    print(feature, ":", missing_count)

    valid_values = [
        row[feature] for row in data
        if row[feature] is not None
    ]

    if not valid_values:
        raise ValueError("Entire column is missing: " + feature)

    column_median = median(valid_values)

    for row in data:
        if row[feature] is None:
            row[feature] = column_median


print("\nMISSING VALUES AFTER IMPUTATION: 0")


# Step 2: Detect and cap outliers using IQR
print("\nOUTLIER DETECTION AND CAPPING")

for feature in features:
    values = [row[feature] for row in data]

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outlier_count = 0

    for row in data:
        value = row[feature]

        if value < lower_limit:
            row[feature] = lower_limit
            outlier_count += 1

        elif value > upper_limit:
            row[feature] = upper_limit
            outlier_count += 1

    print("\nFeature:", feature)
    print("Q1:", round(q1, 4))
    print("Q3:", round(q3, 4))
    print("IQR:", round(iqr, 4))
    print("Lower limit:", round(lower_limit, 4))
    print("Upper limit:", round(upper_limit, 4))
    print("Values capped:", outlier_count)


# Step 3: Z-score normalization
print("\nZ-SCORE NORMALIZATION")

n = len(data)
standardized = []

for feature in features:
    values = [row[feature] for row in data]

    mean = sum(values) / n

    variance = sum(
        (value - mean) ** 2 for value in values
    ) / n

    standard_deviation = math.sqrt(variance)

    print(
        feature,
        "Mean =", round(mean, 4),
        "SD =", round(standard_deviation, 4)
    )

# Each row contains four standardized values
for row in data:
    z_row = []

    for feature in features:
        values = [r[feature] for r in data]
        mean = sum(values) / n

        variance = sum(
            (value - mean) ** 2 for value in values
        ) / n

        sd = math.sqrt(variance)

        if sd == 0:
            z = 0.0
        else:
            z = (row[feature] - mean) / sd

        z_row.append(z)

    standardized.append(z_row)


# Step 4: Calculate the covariance matrix
p = len(features)

covariance = [[0.0] * p for _ in range(p)]

for i in range(p):
    for j in range(p):
        covariance[i][j] = sum(
            standardized[k][i] * standardized[k][j]
            for k in range(n)
        ) / (n - 1)

print("\nCOVARIANCE MATRIX")

for row in covariance:
    print([round(value, 4) for value in row])


# Step 5: Find eigenvalues and eigenvectors
# Jacobi eigenvalue method for a symmetric matrix

def jacobi_eigen(matrix):
    size = len(matrix)

    a = [row[:] for row in matrix]

    # Eigenvectors initially form an identity matrix
    vectors = [
        [1.0 if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]

    for iteration in range(1000):
        max_value = 0.0
        p_index = 0
        q_index = 1

        # Find the largest off-diagonal element
        for i in range(size):
            for j in range(i + 1, size):
                if abs(a[i][j]) > max_value:
                    max_value = abs(a[i][j])
                    p_index = i
                    q_index = j

        if max_value < 1e-10:
            break

        p = p_index
        q = q_index

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        tau = (aqq - app) / (2 * apq)

        if tau >= 0:
            t = 1 / (
                tau + math.sqrt(1 + tau * tau)
            )
        else:
            t = -1 / (
                -tau + math.sqrt(1 + tau * tau)
            )

        c = 1 / math.sqrt(1 + t * t)
        s = t * c

        # Update matrix
        for k in range(size):
            if k != p and k != q:
                akp = a[k][p]
                akq = a[k][q]

                a[k][p] = c * akp - s * akq
                a[p][k] = a[k][p]

                a[k][q] = s * akp + c * akq
                a[q][k] = a[k][q]

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq

        a[p][q] = 0.0
        a[q][p] = 0.0

        # Update eigenvectors
        for k in range(size):
            vkp = vectors[k][p]
            vkq = vectors[k][q]

            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq

    eigenvalues = [a[i][i] for i in range(size)]

    return eigenvalues, vectors


eigenvalues, eigenvectors = jacobi_eigen(covariance)

# Sort eigenvalues in descending order
order = sorted(
    range(p),
    key=lambda i: eigenvalues[i],
    reverse=True
)

eigenvalues = [eigenvalues[i] for i in order]

# Eigenvectors are stored as columns
sorted_vectors = [
    [eigenvectors[row][i] for i in order]
    for row in range(p)
]

total_variance = sum(eigenvalues)

if total_variance <= 0:
    raise ValueError("Total variance must be positive.")

explained_variance = [
    value / total_variance for value in eigenvalues
]

print("\nEIGENVALUES")
print([round(value, 4) for value in eigenvalues])

print("\nEXPLAINED VARIANCE RATIO")
for i, ratio in enumerate(explained_variance):
    print(
        "PC" + str(i + 1),
        "=", round(ratio, 4),
        "or", round(ratio * 100, 2), "%"
    )

print(
    "\nVariance retained by first two PCs:",
    round(sum(explained_variance[:2]) * 100, 2),
    "%"
)

print("\nPCA LOADINGS")

for i, feature in enumerate(features):
    print(
        feature,
        "PC1 =", round(sorted_vectors[i][0], 4),
        "PC2 =", round(sorted_vectors[i][1], 4)
    )


# Step 6: Transform four features into two components
pca_scores = []

for row in standardized:
    pc1 = sum(
        row[i] * sorted_vectors[i][0]
        for i in range(p)
    )

    pc2 = sum(
        row[i] * sorted_vectors[i][1]
        for i in range(p)
    )

    pca_scores.append([pc1, pc2])

print("\nFIRST FIVE PCA RESULTS")

for i in range(min(5, n)):
    print(
        data[i]["Country"],
        "PC1 =", round(pca_scores[i][0], 4),
        "PC2 =", round(pca_scores[i][1], 4)
    )

print("\nOriginal number of features:", p)
print("Reduced number of features: 2")
print("Total records processed:", n)
