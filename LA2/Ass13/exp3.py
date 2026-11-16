
import pandas as pd
import math

df = pd.read_csv("happiness.csv")
df.columns = [str(c).strip() for c in df.columns]


def find_column(candidates):
    for candidate in candidates:
        for col in df.columns:
            if col.lower().strip() == candidate.lower().strip():
                return col
    raise ValueError("Column not found: " + str(candidates))


economy_col = find_column([
    "Economy", "Economy (GDP per Capita)",
    "Logged GDP per capita", "GDP per capita"
])

score_col = find_column([
    "H Score", "Happiness Score", "Score",
    "Happiness.Score", "Ladder score"
])


def get_values(column):
    values = []

    for value in df[column].tolist():
        try:
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        except (ValueError, TypeError):
            pass

    return values


def mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def std_dev(values):
    avg = mean(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return math.sqrt(total / len(values))


def normalize_minmax(values):
    minimum = min(values)
    maximum = max(values)

    result = []

    for value in values:
        if maximum == minimum:
            result.append(0.0)
        else:
            result.append(
                -1 + 2 * (value - minimum) / (maximum - minimum)
            )

    return result


def normalize_zscore(values):
    avg = mean(values)
    sd = std_dev(values)

    if sd == 0:
        return [0.0 for value in values]

    return [(value - avg) / sd for value in values]


def normalize_decimal(values):
    maximum = max(abs(value) for value in values)
    divisor = 1.0

    while maximum / divisor >= 1:
        divisor *= 10

    return [value / divisor for value in values]


def show_stats(name, method, values):
    print(
        name, "|", method,
        "| Min:", round(min(values), 4),
        "| Max:", round(max(values), 4),
        "| Mean:", round(mean(values), 4),
        "| Std:", round(std_dev(values), 4)
    )


for name, column in [
    ("Economy", economy_col),
    ("H Score", score_col)
]:
    original = get_values(column)

    minmax = normalize_minmax(original)
    zscore = normalize_zscore(original)
    decimal = normalize_decimal(original)

    print("\nATTRIBUTE:", name)
    print("Method | Min | Max | Mean | Std")

    show_stats(name, "Original", original)
    show_stats(name, "Min-Max [-1,1]", minmax)
    show_stats(name, "Z-score", zscore)
    show_stats(name, "Decimal Scaling", decimal)

print("\nCONCLUSION")
print(
    "For K-Means, Min-Max normalization is a suitable choice "
    "when features should have a fixed range. Z-score is also "
    "useful when standardizing based on mean and standard deviation."
)
