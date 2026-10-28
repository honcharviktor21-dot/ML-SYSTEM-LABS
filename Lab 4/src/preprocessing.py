import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def engineer_features(X):
    X_eng = X.copy()
    X_eng['capital_net'] = pd.to_numeric(X_eng['capital-gain']) - pd.to_numeric(X_eng['capital-loss'])
    X_eng['hours_per_age'] = pd.to_numeric(X_eng['hours-per-week']) / (pd.to_numeric(X_eng['age']) - 16)
    return X_eng

def clip_outliers(X, percentiles):
    X_clipped = X.copy()
    for col, bounds in percentiles.items():
        X_clipped[col] = np.clip(X_clipped[col], bounds[0], bounds[1])
    return X_clipped

def preprocess_data(df, config):
    print("--- 3. Cleaning & Feature Engineering Step ---")
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['pipeline']['split']['test_size'], 
        stratify=y, random_state=config['pipeline']['split']['random_state']
    )

    train_df = pd.concat([X_train, y_train], axis=1).drop_duplicates()
    X_train, y_train = train_df.drop('income', axis=1), train_df['income']

    null_val = config['pipeline']['schema']['null_policy']
    X_train = X_train.replace(null_val, np.nan)
    X_test = X_test.replace(null_val, np.nan)

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'capital_net', 'hours_per_age']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    train_percentiles = {col: (X_train[col].quantile(0.01), X_train[col].quantile(0.99)) for col in numeric_features}
    clip_transformer = FunctionTransformer(clip_outliers, kw_args={'percentiles': train_percentiles})

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('clipper', clip_transformer), ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(os.path.dirname(config['pipeline']['paths']['preprocessor']), exist_ok=True)
    os.makedirs(os.path.dirname(config['pipeline']['paths']['train_data']), exist_ok=True)

    joblib.dump(preprocessor, config['pipeline']['paths']['preprocessor'])
    
    pd.DataFrame(X_train_processed).assign(income=y_train.values).to_csv(config['pipeline']['paths']['train_data'], index=False)
    pd.DataFrame(X_test_processed).assign(income=y_test.values).to_csv(config['pipeline']['paths']['test_data'], index=False)
    
    print(f"Processed datasets saved to {config['pipeline']['paths']['train_data']} and test_data.")
    print(f"Preprocessor saved to {config['pipeline']['paths']['preprocessor']}\n")