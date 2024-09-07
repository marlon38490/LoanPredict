import pandas as pd

def load_data():
    # Load the dataset
    file_path = "data/Copy of loan.csv"
    data = pd.read_csv(file_path)
    return data


if __name__ == "__main__":
    data = load_data()
    print(data.head())  # Print the first few rows to verify
