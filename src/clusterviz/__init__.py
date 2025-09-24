"""
AgglomerativeViz: Agglomerative clustering toolkit with visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "clusterer team"

# Import main classes for easy access
#from .preprocessing import Preprocessor
from .clusterer import AgglomerativeCluserer
from .evaluation import Evaluator
from .visualize import Visualizer
#from .datasets import load_iris, make_blobs, load_sample_data
#from .report import generate_report

__all__ = [
    "Preprocessor",
    "AgglomerativeCluserer", 
    "Evaluator",
    "Visualizer",
    "load_iris",
    "make_blobs", 
    "load_sample_data",
    "generate_report"
]