"""
Tests for agglomerative clustering components.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from clusterviz.clusterer import AgglomerativeCluserer
from clusterviz.evaluation import Evaluator
from clusterviz.visualize import Visualizer


class TestAgglomerativeClustering:
    """Test cases for agglomerative clustering."""
    
    @pytest.fixture
    def sample_data(self):
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, 
                         random_state=42, cluster_std=1.0)
        return X, y
    
    @pytest.fixture
    def clusterer(self):
        return AgglomerativeCluserer()
    
    @pytest.fixture
    def evaluator(self):
        return Evaluator()
    
    @pytest.fixture
    def visualizer(self):
        return Visualizer()
    
    def test_fit_agglomerative(self, clusterer, sample_data):
        X, _ = sample_data
        
        result = clusterer.fit_agglomerative(X, n_clusters=3, linkage="ward")
        
        assert isinstance(result, dict)
        assert 'labels' in result
        assert 'model' in result
        assert 'params' in result
        assert 'n_clusters' in result
        assert 'algorithm' in result
        
        assert len(result['labels']) == len(X)
        assert len(np.unique(result['labels'])) == 3
        assert result['algorithm'] == 'agglomerative'
    
    def test_different_linkage_methods(self, clusterer, sample_data):
        X, _ = sample_data
        
        linkage_methods = ['ward', 'complete', 'average', 'single']
        
        for linkage in linkage_methods:
            result = clusterer.fit_agglomerative(X, n_clusters=3, linkage=linkage)
            
            assert result['params']['linkage'] == linkage
            assert len(result['labels']) == len(X)
            assert len(np.unique(result['labels'])) == 3
    
    def test_evaluation_metrics(self, evaluator, sample_data):
        """Test evaluation metrics."""
        X, y = sample_data
        
        sil_score = evaluator.silhouette(X, y)
        db_score = evaluator.davies_bouldin(X, y)
        ch_score = evaluator.calinski_harabasz(X, y)
        
        assert isinstance(sil_score, float)
        assert isinstance(db_score, float)
        assert isinstance(ch_score, float)
        
        assert -1 <= sil_score <= 1
        assert db_score >= 0
        assert ch_score > 0
    
    def test_linkage_comparison(self, evaluator, sample_data):
        """Test linkage method comparison."""
        X, _ = sample_data
        
        results_df = evaluator.evaluate_linkage_methods(X, n_clusters=3)
        
        assert len(results_df) == 4  
        assert 'linkage' in results_df.columns
        assert 'silhouette' in results_df.columns
        assert 'davies_bouldin' in results_df.columns
        assert 'calinski_harabasz' in results_df.columns
    
    def test_visualization(self, visualizer, clusterer, sample_data):
        """Test visualization methods."""
        X, _ = sample_data
        result = clusterer.fit_agglomerative(X, n_clusters=3, linkage="ward")
        labels = result['labels']
        ax = visualizer.plot_clusters(X, labels)
        assert ax is not None
        ax = visualizer.plot_dendrogram(X, linkage_method="ward")
        assert ax is not None
        ax = visualizer.plot_silhouette(X, labels)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__])