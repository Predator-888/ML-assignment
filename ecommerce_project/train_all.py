"""
Master Training Script to execute, benchmark, and serialize all models.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.preprocessing import prepare_data
from src.models import train_and_evaluate_all, train_clustering_models
from src.recommend import CollaborativeRecommender

def main():
    print("==================================================")
    print("      E-COMMERCE ML PIPELINE TRAINING START       ")
    print("==================================================")
    
    os.makedirs('ecommerce_project/models', exist_ok=True)
    
    # 1. Prepare Data
    print("\n[Step 1/5] Loading & Preprocessing Dataset...")
    data = prepare_data('ecommerce_project/data/online_shoppers_intention.csv')
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']
    feature_names = data['feature_names']
    raw_df = data['raw_df']
    
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train Shape: {X_train_scaled.shape}, Test Shape: {X_test_scaled.shape}")
    
    # Save Scaler & Features
    joblib.dump(scaler, 'ecommerce_project/models/scaler.pkl')
    joblib.dump(feature_names, 'ecommerce_project/models/feature_names.pkl')
    print("Saved scaler.pkl and feature_names.pkl")

    # 2. Dimensionality Reduction (Exp 5: PCA & t-SNE)
    print("\n[Step 2/5] Training PCA & t-SNE Dimensionality Reduction...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_scaled)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_} (Sum: {pca.explained_variance_ratio_.sum():.4f})")
    joblib.dump(pca, 'ecommerce_project/models/pca_model.pkl')

    # 3. Train Clustering Models (Exp 10: K-Means, Agglomerative, GMM)
    print("\n[Step 3/5] Training Clustering Models (Customer Segmentation)...")
    cluster_df, kmeans_model, gmm_model = train_clustering_models(X_train_scaled, n_clusters=3)
    print("\nClustering Evaluation Benchmark:")
    print(cluster_df.to_string(index=False))
    cluster_df.to_csv('ecommerce_project/models/cluster_benchmark.csv', index=False)
    joblib.dump(kmeans_model, 'ecommerce_project/models/cluster_model.pkl')
    print("Saved cluster_model.pkl (K-Means)")

    # 4. Train Collaborative Filtering (Exp 6: Recommendation Engine)
    print("\n[Step 4/5] Training Collaborative Filtering Recommender...")
    recommender = CollaborativeRecommender()
    recommender.fit_from_df(raw_df, n_samples=300)
    joblib.dump(recommender, 'ecommerce_project/models/recommender.pkl')
    print("Saved recommender.pkl")

    # 5. Train & Benchmark All Classifiers (Exp 4, 7, 8A, 8B, 9)
    print("\n[Step 5/5] Training and Benchmarking 7 Classification Algorithms...")
    results_df, trained_models = train_and_evaluate_all(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    print("\n" + "="*80)
    print("                 CLASSIFIER PERFORMANCE BENCHMARK RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv('ecommerce_project/models/benchmark_results.csv', index=False)
    
    # Select Best Model based on F1-Score (since Revenue is imbalanced)
    best_algo_name = results_df.iloc[0]['Algorithm']
    best_f1 = results_df.iloc[0]['F1-Score']
    best_acc = results_df.iloc[0]['Test Accuracy']
    print("\n" + "="*80)
    print(f">> BEST CLASSIFIER SELECTED: {best_algo_name}")
    print(f"   Accuracy: {best_acc}% | F1-Score: {best_f1}%")
    print("="*80)
    
    best_model = trained_models[best_algo_name]
    
    # If the best model is Keras, save it in h5 or keras format, otherwise joblib
    if 'Keras' in best_algo_name:
        best_model.save('ecommerce_project/models/best_model.keras')
        joblib.dump({'type': 'keras', 'path': 'best_model.keras', 'name': best_algo_name}, 'ecommerce_project/models/best_model_meta.pkl')
    else:
        joblib.dump(best_model, 'ecommerce_project/models/best_model.pkl')
        joblib.dump({'type': 'sklearn', 'name': best_algo_name}, 'ecommerce_project/models/best_model_meta.pkl')
        
    # Also save Random Forest and AdaBoost separately for feature importance inspection
    joblib.dump(trained_models['Random Forest (Tuned)'], 'ecommerce_project/models/rf_model.pkl')
    
    print("\nAll models and artifacts saved successfully in ecommerce_project/models/")

if __name__ == '__main__':
    main()
