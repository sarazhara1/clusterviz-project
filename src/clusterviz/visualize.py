"""
Visualization utilities for agglomerative clustering.

This module provides visualization tools specifically for hierarchical clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples
from typing import Optional, Tuple, List
import warnings


class Visualizer:
    """
    Agglomerative clustering visualization utility.
    
    This class provides methods to create plots for hierarchical clustering.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = 'whitegrid'):
        """
        Initialize the Visualizer.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Default figure size for plots.
        style : str, default='whitegrid'
            Seaborn style for plots.
        """
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
        
    def plot_clusters(self, 
                     X: np.ndarray, 
                     labels: np.ndarray,
                     title: str = "Agglomerative Clustering Results",
                     feature_names: Optional[List[str]] = None,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create a scatter plot of clustering results.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
        title : str
            Title for the plot.
        feature_names : list, optional
            Names of the features for axis labels.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_plot = pca.fit_transform(X)
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        else:
            X_plot = X
            if feature_names and len(feature_names) >= 2:
                xlabel, ylabel = feature_names[0], feature_names[1]
            else:
                xlabel, ylabel = 'Feature 1', 'Feature 2'
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                      c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_dendrogram(self, 
                       X: np.ndarray,
                       linkage_method: str = "ward",
                       max_d: Optional[float] = None,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create a dendrogram for hierarchical clustering.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        linkage_method : str, default="ward"
            The linkage method: 'ward', 'complete', 'average', 'single'.
        max_d : float, optional
            Maximum distance for drawing horizontal line.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        Z = linkage(X, method=linkage_method)
        dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=10)
        
        if max_d is not None:
            ax.axhline(y=max_d, color='red', linestyle='--', alpha=0.7,
                      label=f'Cut at distance: {max_d:.2f}')
            ax.legend()
        
        ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
        ax.set_xlabel('Sample Index or (Cluster Size)')
        ax.set_ylabel('Distance')
        
        return ax
    
    def plot_linkage_comparison(self, 
                               X: np.ndarray,
                               n_clusters: int = 3,
                               linkage_methods: List[str] = None,
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a comparison plot of different linkage methods.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        n_clusters : int, default=3
            Number of clusters to form.
        linkage_methods : list, optional
            List of linkage methods to compare.
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object with comparison plots.
        """
        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average', 'single']
        
        if figsize is None:
            figsize = (15, 4 * ((len(linkage_methods) + 1) // 2))
        
        fig, axes = plt.subplots(nrows=((len(linkage_methods) + 1) // 2), ncols=2, 
                                figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        from .clusterer import AgglomerativeCluserer
        clusterer = AgglomerativeCluserer()
        
        for i, linkage in enumerate(linkage_methods):
            try:
                result = clusterer.fit_agglomerative(X, n_clusters=n_clusters, linkage=linkage)
                labels = result['labels']
                
                title = f"Agglomerative ({linkage} linkage)"
                self.plot_clusters(X, labels, title=title, ax=axes[i])
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error with {linkage}: {str(e)}',
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{linkage} linkage - Error")
        
        for j in range(len(linkage_methods), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette(self, 
                       X: np.ndarray, 
                       labels: np.ndarray,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create a silhouette plot for clustering evaluation.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            ax.text(0.5, 0.5, 'Silhouette analysis requires\nat least 2 clusters',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Silhouette Analysis')
            return ax
        
        sample_silhouette_values = silhouette_samples(X, labels)
        
        y_lower = 10
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i, label in enumerate(unique_labels):
            cluster_silhouette_values = sample_silhouette_values[labels == label]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
            y_lower = y_upper + 10
        
        silhouette_avg = np.mean(sample_silhouette_values)
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        
        ax.set_xlabel('Silhouette Coefficient Values')
        ax.set_ylabel('Cluster Label')
        ax.set_title('Silhouette Analysis for Agglomerative Clustering')
        ax.legend()
        
        return ax