# ClusterViz: Comprehensive Clustering Analysis Toolkit

A polished Python package for clustering analysis that provides multiple clustering algorithms, evaluation metrics, and rich visualizations in a unified, easy-to-use interface.

---

## Project Goals

- **Unified Interface**: Single package for multiple clustering algorithms (KMeans, DBSCAN, Agglomerative, Gaussian Mixture)  
- **Comprehensive Evaluation**: Multiple metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz) with model comparison  
- **Rich Visualizations**: Cluster plots, elbow curves, silhouette analysis, and dendrograms  
- **Educational Focus**: Clear documentation and tutorial notebook for learning clustering concepts  
- **Production Ready**: Test coverage, proper packaging, and modular structure  

---

## Installation

From Source (Recommended for Development):

```bash
# Clone the repository
git clone https://github.com/sarazhara1/clusterviz-project.git
cd clusterviz-project

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
````

### Dependencies

* numpy >= 1.21.0
* pandas >= 1.3.0
* scikit-learn >= 1.0.0
* matplotlib >= 3.5.0
* seaborn >= 0.11.0
* scipy >= 1.7.0

---

## Quick Usage

### Basic Clustering Analysis

```python
import numpy as np
from clusterviz import Preprocessor, Clusterer, Evaluator, Visualizer
from clusterviz.datasets import load_iris, make_blobs_dataset

# Load dataset
X, y = load_iris()

# Preprocess data
preprocessor = Preprocessor()
X_scaled = preprocessor.fit_transform(X, scale=True)

# Fit clustering algorithms
clusterer = Clusterer()
kmeans_result = clusterer.fit_kmeans(X_scaled, k=3)
dbscan_result = clusterer.fit_dbscan(X_scaled, eps=0.5, min_samples=5)

# Evaluate results
evaluator = Evaluator()
print("KMeans Silhouette:", evaluator.silhouette(X_scaled, kmeans_result['labels']))

# Visualize clusters
visualizer = Visualizer()
visualizer.plot_clusters(X_scaled, kmeans_result['labels'], title="KMeans Clustering (k=3)")
```

### Model Selection with Grid Search

```python
k_range = range(2, 8)
grid_results = evaluator.grid_search_kmeans(X_scaled, k_range)

visualizer.plot_elbow(grid_results)
```

---

## API Reference

### Preprocessor

* `fit_transform(X, scale=True, pca_components=None)`
* `transform(X)`
* `get_explained_variance_ratio()`

### Clusterer

* `fit_kmeans(X, k, random_state=42)`
* `fit_dbscan(X, eps, min_samples)`
* `fit_agglomerative(X, n_clusters, linkage="ward")`
* `fit_gmm(X, n_components, covariance_type="full")`

### Evaluator

* `silhouette(X, labels)`
* `davies_bouldin(X, labels)`
* `calinski_harabasz(X, labels)`
* `grid_search_kmeans(X, k_range)`
* `compare_models(results, X)`

### Visualizer

* `plot_clusters(X, labels, title)`
* `plot_elbow(k_results_df)`
* `plot_silhouette(X, labels)`
* `plot_dendrogram(X, linkage="ward")`

---

## Tutorial & Examples

* **[`TUTORIAL.ipynb`](TUTORIAL.ipynb)**: Full walkthrough with Iris and synthetic blob datasets
* Covers:

  * Data loading & preprocessing
  * KMeans with elbow method optimization
  * DBSCAN parameter tuning
  * Agglomerative clustering with dendrograms
  * Model comparison and selection
  * Advanced visualizations

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=clusterviz tests/
```

---

## Contributors

This project was developed as part of the **MSc Data Science coursework at TU Dortmund**.
Each member contributed distinct modules to ensure balanced collaboration:

* **Sara (@sarazhara1)**

  * DBSCAN implementation
  * Davies–Bouldin score evaluation
  * Tutorial notebook (DBSCAN section)

* **Sheryl (@sheryl-bellary)**
  * KMeans implementation
  * Silhouette score evaluation
  * Grid search for KMeans
  * Tutorial notebook (KMeans section)

* **Vidhyaa (@Vidhyaa0309)**

  * Agglomerative clustering
  * Calinski–Harabasz evaluation
  * Visualizations (dendrograms, cluster plots)

Shared effort on:

* Preprocessing utilities
* Dataset loaders
* Overall integration & testing

---

## 📄 License

This project is licensed under the MIT License – see [LICENSE.txt](LICENSE.txt).

---

## 🔍 Model Selection Guide

| Algorithm         | Best For                      | Pros                             | Cons                                            |
| ----------------- | ----------------------------- | -------------------------------- | ----------------------------------------------- |
| **KMeans**        | Spherical clusters, known *k* | Fast, simple, reliable           | Requires *k*, assumes spherical                 |
| **DBSCAN**        | Irregular shapes, unknown *k* | Finds arbitrary shapes, noise    | Sensitive to parameters, varying densities      |
| **Agglomerative** | Hierarchical structure        | No *k* assumption, deterministic | Computationally expensive, sensitive to linkage |
| **GMM**           | Overlapping clusters          | Probabilistic, flexible          | Requires *k*, Gaussian assumption               |

### Evaluation Metrics

* **Silhouette Score (-1 to 1):** Higher = better separation
* **Davies–Bouldin Score (0+):** Lower = better separation
* **Calinski–Harabasz Score (0+):** Higher = better separation

---

**Version:** 1.0.0
**Last Updated:** 2025
**Course:** MSc Data Science – TU Dortmund

```


