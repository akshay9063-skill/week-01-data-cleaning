"""
Week 01 Internship Project
Data Gathering, Cleaning, and Preprocessing
Dataset: Iris flower measurements (UCI Machine Learning Repository)
"""

from pathlib import Path
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    "species",
]

def main():
    # 1. Gather publicly available data
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values=["", "?"})

    print("Initial shape:", df.shape)
    print("\nInitial missing values:\n", df.isna().sum())
    print("\nInitial duplicate rows:", int(df.duplicated().sum()))

    # 2. Validate numerical data types
    numeric_cols = COLUMNS[:4]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Standardise category labels
    df["species"] = (
        df["species"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("iris-", "", regex=False)
    )

    # 4. Remove fully duplicated rows
    df = df.drop_duplicates().copy()

    # 5. Handle missing values where needed
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if df["species"].isna().any():
        mode = df["species"].mode()
        if not mode.empty:
            df["species"] = df["species"].fillna(mode.iloc[0])

    # 6. Save the cleaned data
    output_path = Path(__file__).with_name("cleaned_iris_dataset.csv")
    df.to_csv(output_path, index=False, float_format="%.1f")

    print("\nCleaned shape:", df.shape)
    print("\nCleaned missing values:\n", df.isna().sum())
    print("\nSummary statistics:\n", df[numeric_cols].describe().round(2))
    print(f"\nCleaned dataset saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
