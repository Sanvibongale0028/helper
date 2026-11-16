import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

df = pd.read_csv("iris - iris.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

print(export_text(dt, feature_names=features))

y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
for i, cls in enumerate(dt.classes_):
    print(f"{cls}: Precision={prec[i]:.4f}, Recall={rec[i]:.4f}, F1={f1[i]:.4f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nRoot Attribute:", features[dt.tree_.feature[0]])
print("Reason: It achieves the highest Information Gain (lowest Gini impurity), separating all 'setosa' samples in the first split.")