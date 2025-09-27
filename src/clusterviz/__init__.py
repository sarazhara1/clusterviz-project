"""
ClusterViz: A comprehensive clustering analysis toolkit
"""
__version__ = "1.0.0"
__author__ = "Your Name"

from .preprocessing import Preprocessor
from .clusterer import Clusterer
from .evaluation import Evaluator
from .datasets import load_iris, make_blobs_dataset

__all__ = [
    "Preprocessor",
    "Clusterer", 
    "Evaluator",
    "Visualizer",
    "load_iris",
    "make_blobs_dataset"
]
