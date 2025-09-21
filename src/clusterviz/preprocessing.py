"""
Data preprocessing utilities for clustering.
This module provides a Preprocessor class that can:
Standardize (scale) the data so all features are comparable and 
optionally apply PCA to reduce dimensionality while keeping most variance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, Union


class Preprocessor:
    """
    Data preprocessing for clustering analysis.
    Scaling ensures that features are centered (mean=0) and have unit variance 
(std=1)
    PCA reduces dimensionality by finding new axes that capture maximum variance
    """

    def __init__(self):
        """
        Initializing the preprocessor with empty objects.
        At the start, no scaler or PCA is fitted.
        """
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.is_fitted = False  # Tracks if preprocessing has been fitted on data

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        scale: bool = True,
        pca_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fitting the preprocessing pipeline (scaling + PCA) and applying it to the 
data.
        X denotes the input dataset (numpy array or pandas DataFrame),
        scale (bool) if True, applies StandardScaler
        pca_components (int, optional) If set, applies PCA with this many components
        This function returns the transformed dataset after scaling/PCA (np.ndarray)
        """
        # Ensuring we always work with numpy arrays internally
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Making a copy so original data is not modified
        X_processed = X.copy()

        # Applying standard scaling if requested
        if scale:
            # StandardScaler subtracts the mean and divides by std deviation
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)

        # Applying PCA if requested
        if pca_components is not None:
            # PCA finds directions (principal components) of maximum variance
            # and projects data onto them
            self.pca = PCA(n_components=pca_components, random_state=42)
            X_processed = self.pca.fit_transform(X_processed)

        # Marking that pipeline is fitted
        self.is_fitted = True
        return X_processed

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transforming new data using already fitted scaler/PCA.
        
        One example of its use case is to use the fit_transform() function in train 
data
        and transform() in test data
        The args X denotes the new dataset to be transformed and returns the 
preprocessed
        data in a np.ndarray. It also raises an ValueError if fit_transform() 
function
        has not been called before.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Converting DataFrame in numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_processed = X.copy()

        # Applying same scaling as fitted on training data
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)

        # Applying same PCA projection as fitted
        if self.pca is not None:
            X_processed = self.pca.transform(X_processed)

        return X_processed

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        It gets how much variance each PCA component explains. The returns are:
        np.ndarray if variance ratios (sums to <= 1.0) if PCA was applied or
        None if PCA was not applied
        """
        if self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None

