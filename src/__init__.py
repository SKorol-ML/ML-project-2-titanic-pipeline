"""
Пакет src для проекта Titanic Pipeline.
Содержит модули для предобработки, обучения, оценки и вспомогательных функций.
"""

from .preprocess import create_preprocessor
from .train import get_models_and_params, train_and_select_model, save_model_and_preprocessor
from .evaluate import calculate_metrics, print_metrics_table, plot_confusion_matrix, plot_roc_curve, print_classification_report
from .utils import load_data, split_data, save_predictions

__all__ = [
    'create_preprocessor',
    'get_models_and_params',
    'train_and_select_model',
    'save_model_and_preprocessor',
    'calculate_metrics',
    'print_metrics_table',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'print_classification_report',
    'load_data',
    'split_data',
    'save_predictions',
]