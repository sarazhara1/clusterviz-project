# src/clusterviz/datasets.py
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs
from typing import Tuple, List

def load_iris() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.feature_names

def make_blobs_dataset(n_samples=300, centers=4, n_features=2, 
                       random_state=42, cluster_std=1.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generating a  synthetic blob dataset.
    Returns X, y, feature names.
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
n_features=n_features,
                      random_state=random_state, cluster_std=cluster_std)
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    return X, y, feature_names



def load_customer_data(n_customers: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Generating a  synthetic customer data.
    """
    rng = np.random.default_rng(random_state)
    age = rng.normal(35, 12, n_customers)
    annual_income = rng.normal(50000, 20000, n_customers)
    spending_score = rng.integers(1, 101, n_customers)
    purchase_frequency = rng.poisson(5, n_customers)

    data = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'spending_score': spending_score,
        'purchase_frequency': purchase_frequency
    })
    return data

