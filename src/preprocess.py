import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_csv_path='data/house_Prediction_Data_Set.csv'):
    dataset = pd.read_csv(input_csv_path, sep=r'\s+', header=None)
    
    num_columns = dataset.shape[1]
    column_names = [f'feature_{i}' for i in range(1, num_columns)] + ['target']
    dataset.columns = column_names

    dataset.fillna(dataset.median(numeric_only=True).to_dict(), inplace=True)

    numerical_cols = dataset.columns[:-1]
    scaler = StandardScaler()
    dataset[numerical_cols] = scaler.fit_transform(dataset[numerical_cols])

    X = dataset.drop(columns=['target'])
    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("✅ Data preprocessing completed and saved to /data")

if __name__ == "__main__":
    preprocess_data()
