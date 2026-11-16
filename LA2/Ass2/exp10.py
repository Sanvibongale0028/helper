
import pandas as pd
import random

df = pd.read_csv("iris.csv")
df.columns = df.columns.str.strip()

# Identify the target and four measurement attributes
target_col = next(
    (c for c in df.columns if c.lower() in ["species", "class", "target"]),
    None
)

if target_col is None:
    raise ValueError("Species/target column not found.")

feature_cols = [
    c for c in df.columns
    if c != target_col and c.lower() not in ["id", "index"]
]

if len(feature_cols) != 4:
    raise ValueError("Expected exactly four numeric measurement columns.")

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

X = df[feature_cols].values.tolist()
y = df[target_col].astype(str).tolist()
classes = sorted(set(y))

# 70/30 split with seed 42
indices = list(range(len(X)))
random.seed(42)
random.shuffle(indices)

train_size = int(0.7 * len(indices))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = [X[i] for i in train_indices]
y_train = [y[i] for i in train_indices]
X_test = [X[i] for i in test_indices]
y_test = [y[i] for i in test_indices]

def gini(labels):
    if not labels:
        return 0.0

    total = len(labels)
    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return 1 - sum((count / total) ** 2 for count in counts.values())

def majority_class(labels):
    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return max(counts, key=lambda label: (counts[label], label))

def best_split(X_data, y_data):
    best_gain = 0
    best_result = None
    parent_gini = gini(y_data)

    for feature_index in range(len(feature_cols)):
        values = sorted(set(row[feature_index] for row in X_data))

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left_indices = [
                j for j, row in enumerate(X_data)
                if row[feature_index] <= threshold
            ]
            right_indices = [
                j for j, row in enumerate(X_data)
                if row[feature_index] > threshold
            ]

            if not left_indices or not right_indices:
                continue

            left_labels = [y_data[j] for j in left_indices]
            right_labels = [y_data[j] for j in right_indices]

            weighted_gini = (
                len(left_labels) / len(y_data) * gini(left_labels)
                + len(right_labels) / len(y_data) * gini(right_labels)
            )

            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_result = (
                    feature_index,
                    threshold,
                    left_indices,
                    right_indices
                )

    return best_result

def build_tree(X_data, y_data, depth=0, max_depth=3):
    node = {
        "prediction": majority_class(y_data),
        "samples": len(y_data),
        "gini": gini(y_data),
        "feature": None,
        "threshold": None,
        "left": None,
        "right": None
    }

    if depth >= max_depth or len(set(y_data)) == 1:
        return node

    split = best_split(X_data, y_data)

    if split is None:
        return node

    feature_index, threshold, left_indices, right_indices = split

    node["feature"] = feature_index
    node["threshold"] = threshold

    node["left"] = build_tree(
        [X_data[i] for i in left_indices],
        [y_data[i] for i in left_indices],
        depth + 1,
        max_depth
    )

    node["right"] = build_tree(
        [X_data[i] for i in right_indices],
        [y_data[i] for i in right_indices],
        depth + 1,
        max_depth
    )

    return node

tree = build_tree(X_train, y_train, max_depth=3)

def print_tree(node, depth=0):
    indent = "  " * depth

    if node["feature"] is None:
        print(
            indent + "Leaf: class =", node["prediction"],
            "| samples =", node["samples"],
            "| Gini =", round(node["gini"], 4)
        )
        return

    feature_name = feature_cols[node["feature"]]
    print(
        indent + feature_name + " <= " +
        str(round(node["threshold"], 4)) +
        " | samples = " + str(node["samples"])
    )

    print(indent + "Left:")
    print_tree(node["left"], depth + 1)

    print(indent + "Right:")
    print_tree(node["right"], depth + 1)

print("DECISION TREE")
print_tree(tree)

print("\nRoot attribute:", feature_cols[tree["feature"]]
      if tree["feature"] is not None else "No split")

def predict_one(row, node):
    if node["feature"] is None:
        return node["prediction"]

    if row[node["feature"]] <= node["threshold"]:
        return predict_one(row, node["left"])

    return predict_one(row, node["right"])

predictions = [predict_one(row, tree) for row in X_test]

# Accuracy
correct = sum(
    predictions[i] == y_test[i]
    for i in range(len(y_test))
)
accuracy = correct / len(y_test)

print("\nAccuracy:", round(accuracy, 4))

# Confusion matrix: rows = actual, columns = predicted
matrix = [[0 for _ in classes] for _ in classes]

for actual, predicted in zip(y_test, predictions):
    matrix[classes.index(actual)][classes.index(predicted)] += 1

print("\nConfusion Matrix")
print("Classes:", classes)
for row in matrix:
    print(row)

# Per-class precision, recall and F1
print("\nPer-class metrics")

for i, label in enumerate(classes):
    tp = matrix[i][i]
    fp = sum(matrix[r][i] for r in range(len(classes))) - tp
    fn = sum(matrix[i]) - tp

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(
        label,
        "| Precision:", round(precision, 4),
        "| Recall:", round(recall, 4),
        "| F1-score:", round(f1, 4)
    )
