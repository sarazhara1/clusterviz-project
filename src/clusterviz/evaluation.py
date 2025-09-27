   
"""
Clustering evaluation metrics and model selection utilities.
The Evaluator class provides a consistent interface for model 
assessment and later integration with clusterers.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import  davies_bouldin_score, silhouette_score, calinski_harabasz_score

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
    
    def calinski_harabasz(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz score (Variance Ratio Criterion).
        
        Args:
            X: Input data
            labels: Cluster labels
            
        Returns:
            float: Calinski-Harabasz score (higher is better, ≥0)
        """
        if len(set(labels)) < 2:
            warnings.warn("Calinski-Harabasz score requires at least 2 clusters")
            return 0.0
            
        return calinski_harabasz_score(X, labels)
    
    
    def compare_models(self, results: List[Dict[str, Any]], X: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple clustering results with all metrics.
        
        Args:
            results: List of clustering result dictionaries
            X: Original input data
            
        Returns:
            pd.DataFrame: Comparison table with all metrics
        """
        comparison_data = []
        
        for i, result in enumerate(results):
            labels = result['labels']
            
            # Extract algorithm name and parameters
            if 'k' in result['params']:
                algorithm = 'KMeans'
                param_str = f"k={result['params']['k']}"
            elif 'eps' in result['params']:
                algorithm = 'DBSCAN' 
                param_str = f"eps={result['params']['eps']}, min_samples={result['params']['min_samples']}"
            elif 'n_clusters' in result['params']:
                algorithm = 'Agglomerative'
                param_str = f"n_clusters={result['params']['n_clusters']}, linkage={result['params']['linkage']}"
            elif 'n_components' in result['params']:
                algorithm = 'GMM'
                param_str = f"n_components={result['params']['n_components']}"
            else:
                algorithm = 'Unknown'
                param_str = str(result['params'])
            
            # Calculate metrics
            comparison_data.append({
                'Algorithm': algorithm,
                'Parameters': param_str,
                'N_Clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'Silhouette': self.silhouette(X, labels),
                'Davies_Bouldin': self.davies_bouldin(X, labels),
                'Calinski_Harabasz': self.calinski_harabasz(X, labels),
                'N_Noise_Points': list(labels).count(-1) if -1 in labels else 0
            })
        
        return pd.DataFrame(comparison_data)


    



