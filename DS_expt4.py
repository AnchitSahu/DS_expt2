import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv("cdsp-rainfall.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Information:")
print(df.info())


df = pd.read_csv("cdsp-rainfall.csv")

columns = [
    "average_rainfall",
    "standard_deviation",
    "highest_rainfall",
    "lowest_rainfall"
]

plt.figure(figsize=(12, 10))

for i, col in enumerate(columns):

    data = df[col].dropna()

    upper = data.quantile(0.99)
    data = data[data <= upper]

    plt.subplot(2, 2, i + 1)

    plt.hist(
        data,
        bins=25,
        density=True,
        alpha=0.7,
        rwidth=0.95
    )

    x = np.linspace(data.min(), data.max(), 300)

    bandwidth = 1.06 * np.std(data) * len(data) ** (-1 / 5)

    kde = np.zeros(len(x))

    for value in data.sample(min(3000, len(data)), random_state=1):
        kde += np.exp(-0.5 * ((x - value) / bandwidth) ** 2)

    kde = kde / (
        len(data.sample(min(3000, len(data)), random_state=1))
        * bandwidth
        * np.sqrt(2 * np.pi)
    )

    plt.plot(x, kde, linewidth=2)

    plt.title(col)
    plt.xlabel("Rainfall Value")
    plt.ylabel("Density")

plt.tight_layout()
plt.show()


df = pd.read_csv("cdsp-rainfall(4).csv")

data = df["average_rainfall"].dropna()

minimum = data.min()
maximum = data.max()

scaled = (data - minimum) / (maximum - minimum)

print("Original Minimum:", minimum)
print("Original Maximum:", maximum)

print("\nScaled Minimum:", scaled.min())
print("Scaled Maximum:", scaled.max())

print("\nFirst 10 Scaled Values:")
print(scaled.head(10))


df = pd.read_csv("cdsp-rainfall(4).csv")

data = df[["average_rainfall"]].dropna()

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("First 10 Scaled Values:")
print(scaled_data[:10])

print("\nMinimum:", scaled_data.min())
print("Maximum:", scaled_data.max())


df = pd.read_csv("cdsp-rainfall(4).csv")

selected_codes = [33, 16, 18, 10, 9]

selected = df[df["state_code"].isin(selected_codes)].copy()

encoding = {
    33: 0,
    16: 1,
    18: 2,
    10: 3,
    9: 4
}

selected["state_code_encoded"] = selected["state_code"].map(encoding)

print("State Code Encoding:")

print(
    selected[
        ["state_code", "state_code_encoded"]
    ].drop_duplicates()
)


df = pd.read_csv("cdsp-rainfall(4).csv")

selected_states = [
    "Tamil Nadu",
    "Maharashtra",
    "Gujarat",
    "Kerala",
    "Rajasthan"
]

data = df[df["state_name"].isin(selected_states)].copy()

encoded = pd.get_dummies(
    data["state_name"],
    drop_first=True
)

print("Original Dataset:")
print(data[["state_name"]].head())

print("\nOne-Hot Encoded Data:")
print(encoded.head())

print("\nShape of Encoded Data:")
print(encoded.shape)


df = pd.read_csv("cdsp-rainfall(4).csv")

data = df[["average_rainfall"]].dropna()

minmax = MinMaxScaler()

normalized = minmax.fit_transform(data)

standard = StandardScaler()

standardized = standard.fit_transform(data)

comparison = pd.DataFrame({
    "Original": data["average_rainfall"].values,
    "Normalized": normalized.flatten(),
    "Standardized": standardized.flatten()
})

print("Original vs Transformed Dataset:")
print(comparison.head(10))

print("\n--- Min-Max Scaling ---")
print("Minimum:", normalized.min())
print("Maximum:", normalized.max())

print("\n--- Standardization ---")
print("Mean:", standardized.mean())
print("Standard Deviation:", standardized.std())