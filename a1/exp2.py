import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

# Function to compute five-number summary manually (no describe() or numpy)
def five_number_summary(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    
    # Median
    if n % 2 == 0:
        median = (s[mid - 1] + s[mid]) / 2
        lower_half = s[:mid]
        upper_half = s[mid:]
    else:
        median = s[mid]
        lower_half = s[:mid]
        upper_half = s[mid + 1:]
        
    # Q1 (median of lower half)
    m_low = len(lower_half)
    q1 = (lower_half[m_low // 2 - 1] + lower_half[m_low // 2]) / 2 if m_low % 2 == 0 else lower_half[m_low // 2]
    
    # Q3 (median of upper half)
    m_high = len(upper_half)
    q3 = (upper_half[m_high // 2 - 1] + upper_half[m_high // 2]) / 2 if m_high % 2 == 0 else upper_half[m_high // 2]
    
    return s[0], q1, median, q3, s[-1]

# Compute summary, IQR fences, and outliers for H_Score and Economy
for col in ["H_Score", "Economy"]:
    data = list(df[col])
    minimum, q1, median, q3, maximum = five_number_summary(data)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_fence or x > upper_fence]
    
    print(f"--- {col} ---")
    print(f"Min: {minimum:.4f}, Q1: {q1:.4f}, Median: {median:.4f}, Q3: {q3:.4f}, Max: {maximum:.4f}")
    print(f"IQR Fences: [{lower_fence:.4f}, {upper_fence:.4f}]")
    print(f"Outliers: {outliers}\n")

# Draw box plots
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.boxplot(df["H_Score"])
plt.title("Box Plot of H_Score")
plt.ylabel("Value")

plt.subplot(1, 2, 2)
plt.boxplot(df["Economy"])
plt.title("Box Plot of Economy")

plt.tight_layout()
plt.show()