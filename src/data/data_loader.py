import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataLoader:
    """
    A class for loading and preparing the sentiment analysis data.
    """

    def __init__(self, data_path: Path):
        """
        Initializes the DataLoader with the path to the data.

        Args:
            data_path (Path): The path to the CSV data file.
        """
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the data from the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        return pd.read_csv(self.data_path, encoding='utf-8', delimiter=';')

    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: A tuple containing (X_train, X_test, y_train, y_test).
        """
        df = self.load_data()
        X = df['text']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test