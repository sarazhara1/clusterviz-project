# src/clusterviz/datasets.py
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs
from typing import Tuple, List

def load_iris() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    iris = datasets.load_iris()
    return iris.data, iris.target

def make_blobs_dataset(n_samples=300, centers=4, n_features=2, 
                       random_state=42, cluster_std=1.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generating a  synthetic blob dataset.
    Returns X, y, feature names.
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
                      n_features=n_features,
                      random_state=random_state, cluster_std=cluster_std)
    return X, y, 



def load_customer_data(n_customers) -> pd.DataFrame:
    """
    Generating a  synthetic customer data.
    """
    np.random.seed(42)
    age = np.random.normal(35, 12, n_customers)
    annual_income = np.random.normal(50000, 20000, n_customers)
    spending_score = np.random.randint(1, 101, n_customers)
    purchase_frequency = np.random.poisson(5, n_customers)

    data = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'spending_score': spending_score,
        'purchase_frequency': purchase_frequency
    })
    return pd.DataFrame(data)

