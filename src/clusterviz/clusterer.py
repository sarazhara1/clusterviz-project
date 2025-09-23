import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from typing import Dict, Any

class Clusterer:
    """
    Used for clustering algorithms.
    This class provides consistent interface for Kmeans, DBSCAN, Agglomerative, and 
    GMM.
    """
    
    def __init__(self):
        """Initializing the Clusterer."""
        pass
    
    def fit_dbscan(self, X: np.ndarray, eps: float, min_samples: int) -> Dict[str, Any]:
        """
        Fitting the DBSCAN (Density-Based Spatial Clustering of Applications with Noise)               
        on data. DBSCAN clusters points that are closely packed together, and
        marking as outliers points that lie alone in low-density regions. Unlike KMeans, it 
       does not require specifying the number of clusters in advance.

       Args include:
          X (np.ndarray), which denotes the input data (samples x features)
          eps (float) is the maximum distance between two samples for them to be considered neighbors
          min_samples (int) is the minimum number of points required in a neighborhood to 
          define a core point

       Returns are listed as below:
            dict: Contains clustering results and relevant metadata such as:
           'labels': Cluster labels assigned to each point (-1 means noise)
           'model': The fitted DBSCAN object
           'params': Dictionary of parameters used (eps, min_samples)
           'n_clusters': Number of clusters found (excluding noise)
           'n_noise': Number of noise points (labeled -1)
        """
        # Initializing the DBSCAN model
        model = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Fitting the model and predicting cluster labels
        labels = model.fit_predict(X)
        # fit_predict() function assigns cluster labels to each point
        
        # Calculating the number of clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Counting the number of noise points
        n_noise = list(labels).count(-1)
        
        # Returning a structured dictionary with results and parameters
        return {
            'labels': labels,
            'model': model,
            'params': {'eps': eps, 'min_samples': min_samples},
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
    def fit_kmeans(self, X: np.ndarray, k: int, random_state: int = 42) -> Dict[str, Any]:
        """
        Fit KMeans clustering algorithm.
        
        Args:
            X: Input data
            k (int): Number of clusters
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Results containing labels, model, params, and inertia
        """
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        
        return {
            'labels': labels,
            'model': model,
            'params': {'k': k, 'random_state': random_state},
            'inertia': model.inertia_,
            'cluster_centers': model.cluster_centers_
        }
    
