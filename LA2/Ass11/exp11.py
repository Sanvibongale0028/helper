
import csv
import math

FILE = "weatherHistory.csv"

data = []

# Read actual Kaggle dataset
with open(FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            summary = row["Summary"].strip()
            target = row["Precip Type"].strip().lower()
            temperature = float(row["Temperature (C)"])
            humidity = float(row["Humidity"])
            wind_speed = float(row["Wind Speed (km/h)"])

            if target not in ["rain", "snow"]:
                continue

            if not summary:
                continue

            # Derive categorical Windy feature
            windy = wind_speed >= 20

            data.append((
                summary,
                temperature,
                humidity,
                windy,
                target
            ))

        except (ValueError, KeyError):
            continue


# Select a balanced sample for practical LOOCV
# Select up to 200 records from each target class
def balanced_sample(data, limit):
    rain = [row for row in data if row[4] == "rain"]
    snow = [row for row in data if row[4] == "snow"]

    sample = []

    for group in [rain, snow]:
        size = min(len(group), limit)

        for i in range(size):
            index = i * len(group) // size
            sample.append(group[index])

    return sample


sample = balanced_sample(data, 200)

if not sample:
    raise ValueError("No valid rain/snow records found.")

summaries = sorted(set(row[0] for row in data))
wind_values = [False, True]
classes = ["rain", "snow"]


# Calculate mean and variance
def gaussian_parameters(values):
    mean = sum(values) / len(values)

    variance = sum(
        (x - mean) ** 2 for x in values
    ) / len(values)

    if variance < 1e-9:
        variance = 1e-9

    return mean, variance


# Gaussian log probability
def gaussian_log_probability(x, mean, variance):
    return (
        -0.5 * math.log(2 * math.pi * variance)
        - ((x - mean) ** 2) / (2 * variance)
    )


# Mixed Naive Bayes prediction
def predict(train, test):
    summary, temperature, humidity, windy, _ = test

    scores = {}

    for label in classes:
        rows = [r for r in train if r[4] == label]
        count = len(rows)

        if count == 0:
            scores[label] = float("-inf")
            continue

        # Class prior
        score = math.log(count / len(train))

        # Categorical likelihood: Summary
        matching_summary = sum(
            1 for r in rows if r[0] == summary
        )

        score += math.log(
            (matching_summary + 1) /
            (count + len(summaries))
        )

        # Categorical likelihood: Windy
        matching_wind = sum(
            1 for r in rows if r[3] == windy
        )

        score += math.log(
            (matching_wind + 1) /
            (count + len(wind_values))
        )

        # Gaussian likelihood: Temperature
        temperatures = [r[1] for r in rows]
        mean_t, var_t = gaussian_parameters(temperatures)

        score += gaussian_log_probability(
            temperature, mean_t, var_t
        )

        # Gaussian likelihood: Humidity
        humidities = [r[2] for r in rows]
        mean_h, var_h = gaussian_parameters(humidities)

        score += gaussian_log_probability(
            humidity, mean_h, var_h
        )

        scores[label] = score

    return max(scores, key=scores.get)


# Classify a new day
# Original example: sunny, 66 F, 90%, windy=True
# Kaggle equivalent: Clear, 18.89 C, 0.90 humidity
# Windy=True means wind speed >= 20 km/h

new_day = ("Clear", (66 - 32) * 5 / 9, 0.90, True, "?")

prediction = predict(sample, new_day)

print("NEW DAY PREDICTION")
print("Summary:", new_day[0])
print("Temperature (C):", round(new_day[1], 2))
print("Humidity:", new_day[2])
print("Windy:", new_day[3])
print("Predicted precipitation:", prediction)


# Leave-One-Out Cross-Validation
tp = 0
tn = 0
fp = 0
fn = 0

print("\nLOOCV RESULTS")
print("Actual   Predicted")

for i in range(len(sample)):
    test = sample[i]
    train = sample[:i] + sample[i + 1:]

    actual = test[4]
    predicted = predict(train, test)

    print(actual, "   ", predicted)

    # Positive class = rain
    if actual == "rain" and predicted == "rain":
        tp += 1
    elif actual == "snow" and predicted == "snow":
        tn += 1
    elif actual == "snow" and predicted == "rain":
        fp += 1
    else:
        fn += 1


# Evaluation metrics
total = tp + tn + fp + fn

accuracy = (tp + tn) / total

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0
)

print("\nCONFUSION MATRIX")
print("                 Predicted Rain  Predicted Snow")
print("Actual Rain          ", tp, "             ", fn)
print("Actual Snow          ", fp, "             ", tn)

print("\nEVALUATION METRICS")
print("Records used:", total)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
