import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def add_family_features(df):
    df_copy = df.copy()
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
    return df_copy

def create_preprocessor():
    family_adder = FunctionTransformer(add_family_features, validate=False)
    
    numeric_features = ['Age', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    full_preprocessor = Pipeline(steps=[
        ('add_family', family_adder),
        ('column_transform', column_transformer)
    ])
    
    return full_preprocessor