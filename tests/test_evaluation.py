"""
Tests for evaluation module
"""
import pytest
import numpy as np
import pandas as pd
from clusterviz.evaluation import Evaluator
from clusterviz.datasets import make_blobs_dataset

class TestEvaluator:
    """
    Test suite for the Evaluator class.
    """

    def setup_method(self):
        """
        Creating an Evaluator instance and a synthetic dataset 
        before each test runs.
        - make_blobs_dataset: generates clusterable 2D points
        - X: features
        - y_true: "true" cluster labels (used here just for testing)
        """
        self.evaluator = Evaluator()
        self.X, self.y_true, _ = make_blobs_dataset(
            n_samples=100, 
            centers=3,
            n_features=2
        )

    def test_davies_bouldin_score(self):
        """
        Checking that Evaluator.davies_bouldin:
        - returns a float
        - gives a non-negative score (lower = better clustering)
        """
        score = self.evaluator.davies_bouldin(self.X, self.y_true)

        # Type check
        assert isinstance(score, float)

        # Davies–Bouldin index should always be >= 0
        assert score >= 0


    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        score = self.evaluator.silhouette(self.X, self.y_true)
        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_grid_search_kmeans(self):
        """Test KMeans grid search."""
        results_df = self.evaluator.grid_search_kmeans(self.X, range(2, 6))
        
        assert isinstance(results_df, pd.DataFrame)
        assert 'k' in results_df.columns
        assert 'inertia' in results_df.columns
        assert 'silhouette' in results_df.columns
        assert len(results_df) == 4  # k=2,3,4,5


