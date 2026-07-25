"""
Модуль для обучения и сохранения моделей.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import os

def get_models_and_params():
    """Возвращает словарь моделей и их параметров для GridSearch"""
    models = {
        'LogisticRegression': (LogisticRegression(random_state=42, max_iter=1000), {
            'C': [0.1, 1.0, 10.0],
            'l1_ratio': [0],
            'solver': ['lbfgs']
        }),
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }),
        'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        })
    }
    return models

def train_and_select_model(X_train, y_train, X_test, y_test, models_dict=None):
    """
    Обучает модели с GridSearchCV, возвращает лучшую модель и словарь результатов.
    """
    if models_dict is None:
        models_dict = get_models_and_params()
    
    results = {}
    for name, (model, params) in models_dict.items():
        print(f"\n=== Training {name} ===")
        grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'best_params': grid.best_params_,
            'cv_roc_auc': grid.best_score_,
            'test_roc_auc': test_roc_auc,
            'model': best_model
        }
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    
    # Выбираем лучшую по Test ROC-AUC
    best_name = max(results, key=lambda x: results[x]['test_roc_auc'])
    best_model = results[best_name]['model']
    return best_model, results, best_name

def save_model_and_preprocessor(model, preprocessor, model_name, models_dir='../models'):
    """Сохраняет модель и препроцессор в указанную папку."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, f'{models_dir}/best_model_{model_name}.pkl')
    joblib.dump(preprocessor, f'{models_dir}/preprocessor.pkl')
    print(f"Saved model and preprocessor to {models_dir}/")