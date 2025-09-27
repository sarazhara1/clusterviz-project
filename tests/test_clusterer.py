"""
Tests for clusterer module
"""
import pytest
import numpy as np
from clusterviz.clusterer import Clusterer
from clusterviz.datasets import make_blobs_dataset

class TestClusterer:
    
    def setup_method(self):
        """Setting up test data for clustering."""
        # Initializing the  Clusterer object
        self.clusterer = Clusterer()

        # Generating the blobs synthetic dataset for clustering
        # Blobs: points grouped around centers, commonly used for testing clustering
        self.X_blobs, _, _  = make_blobs_dataset(
            n_samples=100,  # denotes the total number of points
            centers=3,      # number of clusters
        )
    
    def test_dbscan_basic(self):
        """
        Testing basic DBSCAN functionality.
        
        DBSCAN groups points that are closely packed together.
        Parameters include:
            eps which denotes the  maximum distance between points in a neighborhood
            min_samples which is the  minimum number of points required to form a dense region
        Checks include:
             Result contains expected keys
             Number of labels matches input data
             Number of clusters and noise points are reasonable
        """
        # Fitting the  DBSCAN with typical parameters
        result = self.clusterer.fit_dbscan(self.X_blobs, eps=0.5, min_samples=5)
        
       
        # # labels represent the  array of cluster assignments (-1 means noise)
        # # model is the fitted DBSCAN object
        # # params are the  input parameters used for clustering
        # # n_cluster are the number of clusters found (excluding noise)
        # # n_noise represent the  number of points labeled as noise
        # print("DBSCAN labels:", result['labels'])
        # print("Number of clusters:", result['n_clusters'])
        # print("Number of noise points:", result['n_noise'])
        
        # Assertions for testing
        assert 'labels' in result
        assert 'model' in result
        assert 'params' in result
        assert 'n_clusters' in result
        assert 'n_noise' in result
        assert len(result['labels']) == len(self.X_blobs)
        # DBSCAN should identify at least 1 cluster
        assert result['n_clusters'] >= 1
    
    def test_kmeans_basic(self):
        """Test basic KMeans functionality."""
        result = self.clusterer.fit_kmeans(self.X_iris, k=3)
        
        assert 'labels' in result
        assert 'model' in result
        assert 'params' in result
        assert 'inertia' in result
        assert len(result['labels']) == len(self.X_iris)
        assert result['params']['k'] == 3

    def test_agglomerative_basic(self):
        """Test basic Agglomerative functionality."""
        result = self.clusterer.fit_agglomerative(self.X_iris, n_clusters=3)
        
        assert 'labels' in result
        assert 'model' in result
        assert 'params' in result
        assert len(result['labels']) == len(self.X_iris)
        assert result['params']['n_clusters'] == 3

