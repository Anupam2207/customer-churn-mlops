# src/preprocess.py
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def preprocess(data_path, output_train, output_test, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)

    # Drop customerID, handle total charges
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode
    df = pd.get_dummies(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

if __name__ == "__main__":
    data_path = sys.argv[1]
    output_train = sys.argv[2]
    output_test = sys.argv[3]
    test_size = float(sys.argv[4])
    random_state = int(sys.argv[5])

    preprocess(data_path, output_train, output_test, test_size, random_state)
