"""
Agglomerative clustering implementation.

This module provides the AgglomerativeCluserer class for hierarchical clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, Any, Union
import warnings


class AgglomerativeCluserer:
    """
    Agglomerative hierarchical clustering implementation.
    
    This class provides methods to fit agglomerative clustering
    with different linkage criteria.
    """
    
    def __init__(self):
        """Initialize the AgglomerativeCluserer."""
        pass
    
    def fit_agglomerative(self, 
                          X: np.ndarray, 
                          n_clusters: int = 2,
                          linkage: str = "ward",
                          **kwargs) -> Dict[str, Any]:
        """
        Fit Agglomerative Hierarchical clustering algorithm.
        
        Agglomerative clustering recursively merges pairs of clusters
        based on a linkage criterion.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to cluster.
        n_clusters : int, default=2
            The number of clusters to find.
        linkage : str, default="ward"  
            The linkage criterion: 'ward', 'complete', 'average', 'single'.
        **kwargs : dict
            Additional parameters passed to AgglomerativeClustering.
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'labels': cluster labels for each sample
            - 'model': fitted AgglomerativeClustering model
            - 'params': parameters used
            - 'n_clusters': number of clusters
            
        Examples
        --------
        >>> from clusterviz.clusterer import AgglomerativeCluserer
        >>> clusterer = AgglomerativeCluserer()
        >>> result = clusterer.fit_agglomerative(X, n_clusters=4, linkage="complete")
        >>> labels = result['labels']
        """
        # Set parameters
        params = {
            'n_clusters': n_clusters,
            'linkage': linkage,
            **kwargs
        }
        
        # Fit the model
        model = AgglomerativeClustering(**params)
        labels = model.fit_predict(X)
        
        # Prepare result dictionary
        result = {
            'labels': labels,
            'model': model,
            'params': params,
            'n_clusters': n_clusters,
            'algorithm': 'agglomerative'
        }
        
        return result
    
    @staticmethod
    def validate_clustering_input(X: np.ndarray) -> np.ndarray:
        """
        Validate and prepare input data for clustering.
        
        Parameters
        ----------
        X : array-like
            Input data to validate.
            
        Returns  
        -------
        X_validated : np.ndarray
            Validated numpy array.
            
        Raises
        ------
        ValueError
            If input data is invalid for clustering.
        """
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Check for valid shape
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got shape {X.shape}")
        
        if X.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {X.shape[0]}")
        
        # Check for missing values
        if np.any(np.isnan(X)):
            warnings.warn("Input contains NaN values. Consider preprocessing the data.")
        
        return X