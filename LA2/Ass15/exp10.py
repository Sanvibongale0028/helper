
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# Load dataset
df = pd.read_csv("wdbc.data", header=None)

X = []
Y = []

for _, row in df.iterrows():
    try:
        diagnosis = str(row.iloc[1]).strip().upper()
        features = [float(row.iloc[j]) for j in range(2, 32)]

        if diagnosis in ["M", "B"] and all(
            math.isfinite(v) for v in features
        ):
            X.append(features)
            Y.append(1 if diagnosis == "M" else 0)
    except (ValueError, TypeError, IndexError):
        continue

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se",
    "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst",
    "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

print("Total records:", len(X))
print("Total features:", len(feature_names))

# 80/20 train-test split
indices = list(range(len(X)))
random.seed(42)
random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx = indices[:split]
test_idx = indices[split:]

X_train = [X[i] for i in train_idx]
Y_train = [Y[i] for i in train_idx]
X_test = [X[i] for i in test_idx]
Y_test = [Y[i] for i in test_idx]

# Gini impurity
def gini(y):
    if not y:
        return 0

    p = sum(y) / len(y)
    return 1 - p * p - (1 - p) * (1 - p)

# Entropy
def entropy(y):
    if not y:
        return 0

    p = sum(y) / len(y)
    q = 1 - p
    result = 0

    if p > 0:
        result -= p * math.log2(p)

    if q > 0:
        result -= q * math.log2(q)

    return result

# Build decision tree manually
def build_tree(X, Y, criterion, depth=0, max_depth=4):
    positives = sum(Y)

    node = {
        "prediction": 1 if positives >= len(Y) - positives else 0,
        "samples": len(Y),
        "feature": None,
        "threshold": None,
        "left": None,
        "right": None,
        "gain": 0
    }

    if (
        depth >= max_depth or len(set(Y)) <= 1
        or len(Y) < 2
    ):
        return node

    impurity = gini if criterion == "gini" else entropy
    parent = impurity(Y)

    best_gain = 0
    best_feature = None
    best_threshold = None
    best_left = None
    best_right = None

    for feature in range(len(X[0])):
        values = sorted(set(row[feature] for row in X))

        for j in range(len(values) - 1):
            threshold = (values[j] + values[j + 1]) / 2

            left_idx = [
                i for i in range(len(X))
                if X[i][feature] <= threshold
            ]

            right_idx = [
                i for i in range(len(X))
                if X[i][feature] > threshold
            ]

            if not left_idx or not right_idx:
                continue

            left_y = [Y[i] for i in left_idx]
            right_y = [Y[i] for i in right_idx]

            weighted = (
                len(left_y) / len(Y) * impurity(left_y)
                + len(right_y) / len(Y) * impurity(right_y)
            )

            gain = parent - weighted

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left = left_idx
                best_right = right_idx

    if best_feature is None:
        return node

    node["feature"] = best_feature
    node["threshold"] = best_threshold
    node["gain"] = best_gain

    node["left"] = build_tree(
        [X[i] for i in best_left],
        [Y[i] for i in best_left],
        criterion, depth + 1, max_depth
    )

    node["right"] = build_tree(
        [X[i] for i in best_right],
        [Y[i] for i in best_right],
        criterion, depth + 1, max_depth
    )

    return node

# Prediction
def predict(tree, row):
    node = tree

    while node["feature"] is not None:
        if row[node["feature"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]

    return node["prediction"]

# Feature importance
def feature_importances(tree, count, total):
    importance = [0.0] * count

    def visit(node):
        if node["feature"] is None:
            return

        importance[node["feature"]] += (
            node["samples"] / total * node["gain"]
        )

        visit(node["left"])
        visit(node["right"])

    visit(tree)

    total_importance = sum(importance)

    if total_importance > 0:
        importance = [
            x / total_importance for x in importance
        ]

    return importance

# Plot tree
def plot_tree(tree, title):
    positions = {}
    leaf_count = [0]

    def position(node, depth=0):
        if node["feature"] is None:
            x = leaf_count[0]
            leaf_count[0] += 1
        else:
            left = position(node["left"], depth + 1)
            right = position(node["right"], depth + 1)
            x = (left + right) / 2

        positions[id(node)] = (x, -depth)
        return x

    position(tree)

    fig, ax = plt.subplots(figsize=(18, 9))

    def draw(node):
        x, y = positions[id(node)]

        if node["feature"] is None:
            label = (
                "Leaf\n" +
                ("Malignant" if node["prediction"] else "Benign") +
                "\nN=" + str(node["samples"])
            )
        else:
            label = (
                feature_names[node["feature"]] +
                "\n<= " + format(node["threshold"], ".3g") +
                "\nN=" + str(node["samples"])
            )

        ax.text(
            x, y, label, ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
        )

        if node["feature"] is not None:
            for child, branch in [
                (node["left"], "Yes"),
                (node["right"], "No")
            ]:
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [y, cy])
                ax.text((x + cx) / 2, (y + cy) / 2, branch, fontsize=7)
                draw(child)

    draw(tree)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# Evaluation metrics
def evaluate(tree):
    TP = TN = FP = FN = 0

    for row, actual in zip(X_test, Y_test):
        predicted = predict(tree, row)

        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 0:
            TN += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        else:
            FN += 1

    accuracy = (TP + TN) / len(Y_test)
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f3 = (
        10 * precision * recall / (9 * precision + recall)
        if 9 * precision + recall else 0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F3-score:", round(f3, 4))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print([[TN, FP], [FN, TP]])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f3": f3
    }

# Train two trees
gini_tree = build_tree(X_train, Y_train, "gini", max_depth=4)
entropy_tree = build_tree(X_train, Y_train, "entropy", max_depth=4)

# Compare root attributes
print("\nROOT ATTRIBUTE COMPARISON")

for name, tree in [
    ("Gini", gini_tree),
    ("Entropy", entropy_tree)
]:
    if tree["feature"] is not None:
        print(
            name, ":",
            feature_names[tree["feature"]],
            "threshold =", round(tree["threshold"], 4)
        )
    else:
        print(name, ": no split")

# Plot both trees
plot_tree(gini_tree, "Decision Tree - Gini")
plot_tree(entropy_tree, "Decision Tree - Entropy")

# Top five feature importances
print("\nTOP FIVE FEATURE IMPORTANCES")

for name, tree in [
    ("Gini", gini_tree),
    ("Entropy", entropy_tree)
]:
    importance = feature_importances(
        tree, len(feature_names), len(X_train)
    )

    order = sorted(
        range(len(importance)),
        key=lambda i: importance[i],
        reverse=True
    )

    print("\n", name)

    for i in order[:5]:
        print(feature_names[i], round(importance[i], 5))

# Evaluate both trees
print("\nGINI TREE EVALUATION")
gini_results = evaluate(gini_tree)

print("\nENTROPY TREE EVALUATION")
entropy_results = evaluate(entropy_tree)

# Prefer recall first, then F3-score
if (
    gini_results["recall"], gini_results["f3"]
) >= (
    entropy_results["recall"], entropy_results["f3"]
):
    print("\nPreferred tree: Gini")
else:
    print("\nPreferred tree: Entropy")

print(
    "Reason: Recall for malignant cases is important because "
    "a false negative means a malignant case is predicted as benign."
)
