"""
Вспомогательные функции для загрузки данных, разделения выборок и сохранения результатов.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Загружает данные из CSV-файла и возвращает DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден")
    return pd.read_csv(filepath)

def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Разделяет данные на обучающую и тестовую выборки с возможностью стратификации.
    """
    if stratify:
        strat = y
    else:
        strat = None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

def save_predictions(y_pred, filepath='../predictions.csv', ids=None):
    """
    Сохраняет предсказания в CSV-файл.
    """
    if ids is None:
        ids = np.arange(len(y_pred))
    df_out = pd.DataFrame({'PassengerId': ids, 'Survived': y_pred})
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_out.to_csv(filepath, index=False)
    print(f"Предсказания сохранены в {filepath}")