import numpy as np
import pandas as pd
import pytest
from clusterviz.preprocessing import Preprocessor


def test_fit_transform_scaling_only():
    """Checking that scaling centers data to mean≈0 and std≈1."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    pre = Preprocessor()
    X_scaled = pre.fit_transform(X, scale=True, pca_components=None)

    # After scaling, mean should be 0 and std 1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)


def test_fit_transform_with_pca():
    """Checking if PCA reduces dimensionality and explains variance."""
    X = np.random.rand(10, 5)
    pre = Preprocessor()
    X_pca = pre.fit_transform(X, scale=True, pca_components=2)

    # Shape check (10 samples, 2 components)
    assert X_pca.shape == (10, 2)

    # Variance ratio should exist and sum <= 1
    variance_ratio = pre.get_explained_variance_ratio()
    assert variance_ratio is not None
    assert variance_ratio.sum() <= 1.0
    assert variance_ratio.sum() > 0.0


def test_transform_reuses_fitted_scaler_pca():
    """Checking that transform() uses the same fitted scaler and PCA."""
    X_train = np.random.rand(10, 5)
    X_test = np.random.rand(5, 5)

    pre = Preprocessor()
    pre.fit_transform(X_train, scale=True, pca_components=2)
    X_test_transformed = pre.transform(X_test)

    # Test set must be transformed to 2 dimensions as well
    assert X_test_transformed.shape == (5, 2)


def test_transform_without_fit_raises_error():
    """Ensuring error is raised if transform is called before fit."""
    X = np.random.rand(5, 3)
    pre = Preprocessor()
    with pytest.raises(ValueError):
        pre.transform(X)


def test_fit_transform_with_dataframe():
    """Checking if DataFrame input also works."""
    df = pd.DataFrame(np.random.rand(8, 4))
    pre = Preprocessor()
    X_out = pre.fit_transform(df, scale=True, pca_components=2)

    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == (8, 2)

