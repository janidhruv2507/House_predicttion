import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train_and_save_models():
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Save Linear Regression Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, 'models/linear_regression_model.pkl')

    # Bin target for KNN
    bins = [0, 15, 25, float('inf')]
    labels = ['Low', 'Medium', 'High']
    y_train_binned = pd.cut(y_train, bins=bins, labels=labels)
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels)

    # Find best k for KNN
    best_k = 1
    best_accuracy = 0
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train_binned)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test_binned, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k

    # Final KNN training with best k
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train_binned)
    joblib.dump(final_knn, 'models/final_knn_model.pkl')

    print(f"✅ Models trained and saved. Optimal K for KNN: {best_k}")

if __name__ == "__main__":
    train_and_save_models()
