"""
Модуль для оценки моделей: метрики, матрица ошибок, ROC-кривая.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Вычисляет основные метрики классификации.
    Возвращает словарь с метриками.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def print_metrics_table(metrics):
    """Выводит таблицу метрик в консоль."""
    df = pd.DataFrame([metrics])
    print("=== Основные метрики ===")
    print(df.round(4))

def plot_confusion_matrix(y_true, y_pred, labels=['Погиб', 'Выжил'], figsize=(6,5), cmap='Blues'):
    """
    Строит и отображает матрицу ошибок.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    plt.title('Матрица ошибок')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name='Модель', figsize=(8,6)):
    """
    Строит ROC-кривую и выводит AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return auc


def print_classification_report(y_true, y_pred, target_names=['Погиб', 'Выжил']):
    """Выводит полный отчёт классификации."""
    print("\n=== Детальный отчёт классификации ===")
    print(classification_report(y_true, y_pred, target_names=target_names))