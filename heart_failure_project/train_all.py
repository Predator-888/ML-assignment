"""
Master Training Script for Heart Failure Clinical ML Project.
Executes, benchmarks, and serializes all models across Experiments 4-10.
"""

import os
import sys

# Ensure heart_failure_project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.preprocessing import prepare_data
from src.models import train_and_evaluate_all, train_clustering_models
from src.recommend import ClinicalCollaborativeRecommender

def main():
    print("==================================================")
    print("   HEART FAILURE CLINICAL ML PIPELINE START       ")
    print("==================================================")
    
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Prepare Data
    print("\n[Step 1/5] Loading & Preprocessing Clinical Dataset...")
    data_path = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records.csv')
    data = prepare_data(data_path)
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']
    feature_names = data['feature_names']
    raw_df = data['raw_df']
    
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train Cohort: {X_train_scaled.shape}, Test Cohort: {X_test_scaled.shape}")
    
    # Save Scaler & Features
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    print("Saved scaler.pkl and feature_names.pkl")

    # 2. Dimensionality Reduction (Exp 5: PCA & t-SNE)
    print("\n[Step 2/5] Training PCA & t-SNE Dimensionality Reduction...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_scaled)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_} (Sum: {pca.explained_variance_ratio_.sum():.4f})")
    joblib.dump(pca, os.path.join(models_dir, 'pca_model.pkl'))

    # 3. Train Clustering Models (Exp 10: K-Means, Agglomerative, GMM)
    print("\n[Step 3/5] Training Clustering Models (Patient Risk Stratification)...")
    cluster_df, kmeans_model, gmm_model = train_clustering_models(X_train_scaled, n_clusters=3)
    print("\nClustering Evaluation Benchmark:")
    print(cluster_df.to_string(index=False))
    cluster_df.to_csv(os.path.join(models_dir, 'cluster_benchmark.csv'), index=False)
    joblib.dump(kmeans_model, os.path.join(models_dir, 'cluster_model.pkl'))
    print("Saved cluster_model.pkl (K-Means)")

    # 4. Train Collaborative Filtering (Exp 6: Clinical Recommender)
    print("\n[Step 4/5] Training Collaborative Filtering Recommender...")
    recommender = ClinicalCollaborativeRecommender()
    recommender.fit_from_df(raw_df)
    joblib.dump(recommender, os.path.join(models_dir, 'recommender.pkl'))
    print("Saved recommender.pkl")

    # 5. Train & Benchmark All Classifiers (Exp 4, 7, 8A, 8B, 9)
    print("\n[Step 5/5] Training and Benchmarking 7 Classification Algorithms...")
    results_df, trained_models = train_and_evaluate_all(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    print("\n" + "="*80)
    print("          HEART FAILURE CLASSIFIER PERFORMANCE BENCHMARK RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(models_dir, 'benchmark_results.csv'), index=False)
    
    # Select Best Model based on F1-Score
    best_algo_name = results_df.iloc[0]['Algorithm']
    best_f1 = results_df.iloc[0]['F1-Score']
    best_acc = results_df.iloc[0]['Test Accuracy']
    print("\n" + "="*80)
    print(f">> BEST CLASSIFIER SELECTED: {best_algo_name}")
    print(f"   Accuracy: {best_acc}% | F1-Score: {best_f1}%")
    print("="*80)
    
    best_model = trained_models[best_algo_name]
    
    if 'Keras' in best_algo_name:
        best_model.save(os.path.join(models_dir, 'best_model.keras'))
        joblib.dump({'type': 'keras', 'path': 'best_model.keras', 'name': best_algo_name}, os.path.join(models_dir, 'best_model_meta.pkl'))
    else:
        joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
        joblib.dump({'type': 'sklearn', 'name': best_algo_name}, os.path.join(models_dir, 'best_model_meta.pkl'))
        
    joblib.dump(trained_models['Random Forest (Tuned)'], os.path.join(models_dir, 'rf_model.pkl'))
    print("\nAll clinical models and artifacts saved successfully in heart_failure_project/models/")

if __name__ == '__main__':
    main()

