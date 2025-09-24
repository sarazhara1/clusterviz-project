"""
Clustering evaluation metrics for agglomerative clustering.

This module provides evaluation metrics specifically for agglomerative clustering.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .clusterer import AgglomerativeCluserer
from typing import List, Dict, Any
import warnings


class Evaluator:
    """
    Clustering evaluation utility for agglomerative clustering.
    
    This class provides methods to evaluate agglomerative clustering results.
    """
    
    def __init__(self):
        """Initialize the Evaluator."""
        self.clusterer = AgglomerativeCluserer()
    
    def silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the Silhouette Coefficient for clustering evaluation.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data used for clustering.
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
            
        Returns
        -------
        score : float
            The mean Silhouette Coefficient.
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            warnings.warn("Silhouette score undefined for single cluster.")
            return np.nan
            
        try:
            return silhouette_score(X, labels)
        except Exception as e:
            warnings.warn(f"Error computing silhouette score: {e}")
            return np.nan
    
    def davies_bouldin(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the Davies-Bouldin Index for clustering evaluation.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data used for clustering.
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
            
        Returns
        -------
        score : float
            The Davies-Bouldin Index.
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            warnings.warn("Davies-Bouldin score undefined for single cluster.")
            return np.nan
            
        try:
            return davies_bouldin_score(X, labels)
        except Exception as e:
            warnings.warn(f"Error computing Davies-Bouldin score: {e}")
            return np.nan
    
    def calinski_harabasz(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the Calinski-Harabasz Index for clustering evaluation.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data used for clustering.
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
            
        Returns
        -------
        score : float
            The Calinski-Harabasz Index.
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            warnings.warn("Calinski-Harabasz score undefined for single cluster.")
            return np.nan
            
        try:
            return calinski_harabasz_score(X, labels)
        except Exception as e:
            warnings.warn(f"Error computing Calinski-Harabasz score: {e}")
            return np.nan
    
    def evaluate_linkage_methods(self, 
                                X: np.ndarray, 
                                n_clusters: int = 3,
                                linkage_methods: List[str] = None) -> pd.DataFrame:
        """
        Compare different linkage methods for agglomerative clustering.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data for clustering.
        n_clusters : int, default=3
            Number of clusters to form.
        linkage_methods : list, optional
            List of linkage methods to compare.
            
        Returns
        -------
        results_df : pd.DataFrame
            DataFrame comparing linkage methods with metrics.
        """
        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average', 'single']
        
        results = []
        
        for linkage in linkage_methods:
            try:
                result = self.clusterer.fit_agglomerative(X, n_clusters=n_clusters, linkage=linkage)
                labels = result['labels']
                metrics = {
                    'linkage': linkage,
                    'n_clusters': n_clusters,
                    'silhouette': self.silhouette(X, labels),
                    'davies_bouldin': self.davies_bouldin(X, labels),
                    'calinski_harabasz': self.calinski_harabasz(X, labels)
                }
                
                results.append(metrics)
                
            except Exception as e:
                warnings.warn(f"Error with {linkage} linkage: {e}")
                continue
        
        return pd.DataFrame(results)