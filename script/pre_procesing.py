import pandas as pd

def prep_reviews(columns: list = []) -> pd.DataFrame:
    """
    Preprocesses the data by performing the following steps:
    1. Retrieves the raw dataset using the specified columns.
    2. Drops any duplicate rows.
    3. Drops any rows with missing values.
    4. Converts the timestamp column to datetime format.
    5. Sorts the dataset by timestamp.
    6. Saves the preprocessed data to a CSV file.

    Args:
        columns (list, optional): List of column names to retrieve from the raw dataset. Defaults to [].

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    import os
    from data_gathering import get_raw_reviews
    os.makedirs("data/_processed/") if not os.path.exists("data/_processed/") else None
    df = get_raw_reviews(columns=columns, toDF=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop rows with missing values
    df.dropna(inplace=True)
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
    # Sort by timestamp
    df = df.sort_values(by="timestamp")
    # Save the preprocessed data
    df.to_csv("data/_processed/reviews.csv", index=False)
    return df

def main():
    prep_reviews(['rating', 'parent_asin', 'user_id', 'timestamp'])

if __name__ == "__main__":
    main()