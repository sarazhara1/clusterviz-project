   
"""
Clustering evaluation metrics and model selection utilities.
The Evaluator class provides a consistent interface for model 
assessment and later integration with clusterers.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import  davies_bouldin_score, silhouette_score

from .clusterer import Clusterer
from typing import List, Dict, Any
import warnings

class Evaluator:
    """
    Clustering evaluation and model selection utilities.    
    Provides metrics, calculation and grid search capabilities.

    """
    
    def __init__(self):
        # Keep a reference to the Clusterer class in case we want 
        # to fit models and evaluate them in one place.
        self.clusterer = Clusterer()
    
    def davies_bouldin(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculating the Davies–Bouldin index for a clustering result.
        Parameters include:
            X: np.ndarray 
                The feature matrix (rows = samples, columns = features).
            labels : np.ndarray
            Cluster assignments for each sample.
            
        Returns:
            float:  The Davies–Bouldin score. Lower values indicate better clustering.
            If there are fewer than 2 clusters, returns `inf`.
        """
        # DB score is undefined if everything is in one cluster
        if len(set(labels)) < 2:
            warnings.warn("Davies-Bouldin score requires at least 2 clusters")
            return float('inf')
            
        return davies_bouldin_score(X, labels)
    

    def silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score.
        
        Args:
            X: Input data
            labels: Cluster labels
            
        Returns:
            float: Silhouette score (-1 to 1, higher is better)
        """
        # Check for minimum requirements
        n_unique_labels = len(set(labels))
        if n_unique_labels < 2 or n_unique_labels >= len(X):
            warnings.warn("Silhouette score requires at least 2 clusters and fewer than n_samples clusters")
            return -1.0
        
        return silhouette_score(X, labels)
    
    def grid_search_kmeans(self, X: np.ndarray, k_range: range, 
                          random_state: int = 42) -> pd.DataFrame:
        """
        Perform grid search over KMeans k values with multiple metrics.
        
        Args:
            X: Input data
            k_range: Range of k values to test
            random_state: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Results with k, inertia, silhouette, davies_bouldin, calinski_harabasz
        """
        results = []
        
        for k in k_range:
            # Fit KMeans
            kmeans_result = self.clusterer.fit_kmeans(X, k, random_state)
            labels = kmeans_result['labels']
            inertia = kmeans_result['inertia']
            
            # Calculate metrics
            sil_score = self.silhouette(X, labels) if k > 1 else -1
            db_score = self.davies_bouldin(X, labels) if k > 1 else float('inf')
            ch_score = self.calinski_harabasz(X, labels) if k > 1 else 0
            
            results.append({
                'k': k,
                'inertia': inertia,
                'silhouette': sil_score,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score
            })
        
        return pd.DataFrame(results)


    



