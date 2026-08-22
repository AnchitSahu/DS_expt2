import pandas as pd
import numpy as np

df = pd.read_csv("cdsp-rainfall.csv")

df.info()
print(df.head(20))
print("Size of the file:", df.shape)
print("\nDescription of the dataset:")
print(df.describe())
print("\nNull values before imputation:")
print(df.isnull().sum())

columns = [
    "average_rainfall",
    "standard_deviation",
    "highest_rainfall",
    "lowest_rainfall"
]

for column in columns:
    mean = df[column].mean()
    median = df[column].median()

    print("\nColumn:", column)
    print("Mean:", mean)
    print("Median:", median)
    print("Difference:", abs(mean - median))

df.fillna(
    {
        "average_rainfall": df["average_rainfall"].median(),
        "standard_deviation": df["standard_deviation"].median(),
        "highest_rainfall": df["highest_rainfall"].median(),
        "lowest_rainfall": df["lowest_rainfall"].median()
    },
    inplace=True
)

print("\nNull values after imputation:")
print(df.isnull().sum())

print(df.sort_values(
    by=["average_rainfall", "highest_rainfall"],
    ascending=[False, False]
).head(8))
filtered_data = df[df["average_rainfall"] > 4700]
print(filtered_data.sort_values(
    by="average_rainfall",
    ascending=False
).head(8))

print("Frequency of states:")
print(df["state_name"].value_counts())
print("Frequency of districts:")
print(df["district_name"].value_counts())

print("Sorted Rows:")
print(df.sort_values(by="average_rainfall", ascending=False).head(8))
print("\nSorted Columns:")
print(df[sorted(df.columns)])
print("\nImplicit Indexing:")
print(df.iloc[0:5, 0:4])
print("\nExplicit Indexing:")
print(df.loc[0:4, ["date", "state_name", "average_rainfall"]])

print("Case 1:")
print(df.loc[
    (df["average_rainfall"] > 100) & (df["state_name"] == "Tamil Nadu"),
    ["date", "district_name", "average_rainfall"]
].head(8))
print("\nCase 2:")
print(df.loc[
    (df["highest_rainfall"] > 200) & (df["average_rainfall"] > 100),
    ["state_name", "station_name", "highest_rainfall"]
].head(8))
print("\nCase 3:")
print(df.loc[
    (df["lowest_rainfall"] < 30) & (df["average_rainfall"] > 90),
    ["date", "state_name", "lowest_rainfall"]
].head(8))

print("Minimum values:")
print("Average Rainfall:", df["average_rainfall"].min())
print("Standard Deviation:", df["standard_deviation"].min())
print("Highest Rainfall:", df["highest_rainfall"].min())
print("Lowest Rainfall:", df["lowest_rainfall"].min())
print("\nMaximum values:")
print("Average Rainfall:", df["average_rainfall"].max())
print("Standard Deviation:", df["standard_deviation"].max())
print("Highest Rainfall:", df["highest_rainfall"].max())
print("Lowest Rainfall:", df["lowest_rainfall"].max())

print("Average rainfall by state:")
print(df.groupby("state_name")["average_rainfall"].mean())
print("\nAverage rainfall by state and district:")
print(df.groupby(["state_name", "district_name"])["average_rainfall"].mean())

df["rainfall_range"] = df["highest_rainfall"] - df["lowest_rainfall"]
print(df[["highest_rainfall", "lowest_rainfall", "rainfall_range"]].head(20))

print("State-wise rainfall analysis:")
print(df.groupby("state_name").agg({
"average_rainfall": "mean",
"highest_rainfall": "max",
"lowest_rainfall": "min"
}))
print("\nDistrict-wise rainfall analysis:")
print(df.groupby("district_name").agg({
    "average_rainfall": "mean",
    "highest_rainfall": "max",
    "lowest_rainfall": "min"
}))

tamil_nadu = df[df["state_name"] == "Tamil Nadu"]
print(tamil_nadu.head(8))
state_rainfall = df.groupby("state_name")["average_rainfall"].mean()
print(state_rainfall[state_rainfall > 100])

correlation = df["average_rainfall"].corr(df["highest_rainfall"])
print("Correlation between average rainfall and highest rainfall:", correlation)

minimum = df["average_rainfall"].min()
maximum = df["average_rainfall"].max()
df["average_rainfall_normalized"] = (
    (df["average_rainfall"] - minimum) /
    (maximum - minimum)
)
print(df[["average_rainfall", "average_rainfall_normalized"]].head(8))

df1 = df[["station_code", "station_name"]].drop_duplicates().head(10)
df2 = df[["station_code", "average_rainfall"]].drop_duplicates().head(10)
joined_data = df1.join(df2.set_index("station_code"), on="station_code")
print("Joined data:")
print(joined_data)

df1 = df[["station_code", "station_name"]].drop_duplicates().head(10)
df2 = df[["station_code", "average_rainfall"]].drop_duplicates().head(10)
merged_data = pd.merge(df1, df2, on="station_code")
print("\nMerged data:")
print(merged_data)

part1 = df.head(5)
part2 = df.iloc[5:10]
concatenated_data = pd.concat([part1, part2])
print("\nConcatenated data:")
print(concatenated_data)








