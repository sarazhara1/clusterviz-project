"""
Clustering evaluation metrics and model selection utilities.
The Evaluator class provides a consistent interface for model 
assessment and later integration with clusterers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import  davies_bouldin_score
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
    
    



