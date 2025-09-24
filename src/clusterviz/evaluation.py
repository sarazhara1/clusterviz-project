"""
Clustering evaluation metrics and model selection utilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .clusterer import Clusterer
from typing import List, Dict, Any
import warnings

class Evaluator:
    """
    Clustering evaluation and model selection utilities.
    
    Provides metrics calculation and grid search capabilities.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.clusterer = Clusterer()
    
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