"""
Data Preprocessing and Clinical Feature Engineering Module for Heart Failure Project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath='heart_failure_project/data/heart_failure_clinical_records.csv'):
    """Load the heart failure clinical dataset."""
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    """Create clinically relevant engineered interaction features."""
    df = df.copy()
    
    # 1. Cardiorenal Risk Ratio: serum creatinine vs ejection fraction
    df['Renal_Cardiac_Ratio'] = df['serum_creatinine'] / (df['ejection_fraction'] + 1e-5)
    
    # 2. Hyponatremia Risk Flag: serum sodium < 135 mEq/L
    df['Hyponatremia_Flag'] = (df['serum_sodium'] < 135).astype(int)
    
    # 3. Severe Systolic Dysfunction Flag: Ejection fraction < 30%
    df['Severe_EF_Flag'] = (df['ejection_fraction'] < 30).astype(int)
    
    # 4. Geriatric Risk Indicator: Age >= 65
    df['Age_Risk_Group'] = (df['age'] >= 65).astype(int)
    
    # 5. Elevated Renal Biomarker Flag: serum creatinine > 1.4 mg/dL
    df['High_Creatinine_Flag'] = (df['serum_creatinine'] > 1.4).astype(int)
    
    # 6. Comorbidity Multiplier: sum of chronic risk factors
    df['Comorbidity_Count'] = (
        df['anaemia'] + 
        df['diabetes'] + 
        df['high_blood_pressure'] + 
        df['smoking']
    )
    
    return df

def prepare_data(filepath='heart_failure_project/data/heart_failure_clinical_records.csv', test_size=0.2, random_state=42):
    """
    Complete pipeline: load, engineer, stratified split, and scale features.
    Returns:
        dict containing train/test data, scaler, feature names, and raw DataFrame.
    """
    raw_df = load_data(filepath)
    fe_df = engineer_features(raw_df)
    
    target_col = 'DEATH_EVENT'
    feature_cols = [col for col in fe_df.columns if col != target_col]
    
    X = fe_df[feature_cols]
    y = fe_df[target_col]
    
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
        'engineered_df': fe_df
    }

def transform_single_patient(patient_dict, scaler, feature_names):
    """Transform a single patient record dictionary into scaled array for live inference."""
    df = pd.DataFrame([patient_dict])
    df = engineer_features(df)
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    
    scaled_array = scaler.transform(df)
    return scaled_array, df

