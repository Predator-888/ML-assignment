"""
Data Preprocessing and Feature Engineering Module for E-Commerce Project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

VISITOR_MAP = {
    'Returning_Visitor': 0,
    'New_Visitor': 1,
    'Other': 2
}

def load_data(filepath='ecommerce_project/data/online_shoppers_intention.csv'):
    """Load the e-commerce dataset."""
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    """Create domain-specific engineered features."""
    df = df.copy()
    
    # 1. Total page views across categories
    df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    
    # 2. Total duration spent on site (seconds)
    df['TotalDuration'] = (
        df['Administrative_Duration'] + 
        df['Informational_Duration'] + 
        df['ProductRelated_Duration']
    )
    
    # 3. Bounce to Exit ratio
    df['BounceExitRatio'] = df['BounceRates'] / (df['ExitRates'] + 1e-5)
    
    # 4. Average duration per page visited
    df['AvgPageDuration'] = df['TotalDuration'] / (df['TotalPages'] + 1e-5)
    
    # 5. Ratio of product pages vs total pages
    df['ProductPageRatio'] = df['ProductRelated'] / (df['TotalPages'] + 1e-5)
    
    # 6. Ratio of product duration vs total duration
    df['ProductDurationRatio'] = df['ProductRelated_Duration'] / (df['TotalDuration'] + 1e-5)
    
    return df

def encode_categorical(df):
    """Encode categorical and boolean variables."""
    df = df.copy()
    
    # Month mapping
    if 'Month' in df.columns:
        df['Month_Num'] = df['Month'].map(MONTH_MAP).fillna(0).astype(int)
        df = df.drop(columns=['Month'])
        
    # VisitorType mapping
    if 'VisitorType' in df.columns:
        df['VisitorType_Code'] = df['VisitorType'].map(VISITOR_MAP).fillna(2).astype(int)
        df = df.drop(columns=['VisitorType'])
        
    # Weekend
    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)
        
    # Revenue (Target)
    if 'Revenue' in df.columns:
        df['Revenue'] = df['Revenue'].astype(int)
        
    return df

def prepare_data(filepath='ecommerce_project/data/online_shoppers_intention.csv', test_size=0.2, random_state=42):
    """
    Full pipeline: load, engineer, encode, split, and scale data.
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names, raw_df
    """
    raw_df = load_data(filepath)
    fe_df = engineer_features(raw_df)
    encoded_df = encode_categorical(fe_df)
    
    target_col = 'Revenue'
    feature_cols = [col for col in encoded_df.columns if col != target_col]
    
    X = encoded_df[feature_cols]
    y = encoded_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_cols,
        'raw_df': raw_df,
        'encoded_df': encoded_df
    }

def transform_single_input(input_dict, scaler, feature_names):
    """Transform a single dictionary input into scaled array for prediction."""
    df = pd.DataFrame([input_dict])
    df = engineer_features(df)
    df = encode_categorical(df)
    
    # Ensure all required features are present and aligned
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    
    scaled_array = scaler.transform(df)
    return scaled_array, df

