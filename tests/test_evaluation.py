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
