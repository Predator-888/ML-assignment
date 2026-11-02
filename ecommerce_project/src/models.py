"""
Model Training, Comparison, and Evaluation Module.
Directly models algorithms from:
- Exp 4: K-Nearest Neighbors (KNN)
- Exp 5: PCA & t-SNE
- Exp 7: SVM (Linear & RBF)
- Exp 8A: Multi-Layer Perceptron (MLP)
- Exp 8B: Deep Neural Network (TensorFlow/Keras)
- Exp 9: Random Forest & AdaBoost
- Exp 10: K-Means, Agglomerative, GMM Clustering
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Train and benchmark all 7 classifiers from lab experiments.
    Returns:
        results_df: DataFrame with comparative metrics
        trained_models: Dictionary of trained model instances
    """
    models = {
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'SVM (Linear Kernel)': SVC(kernel='linear', C=1.0, probability=True, max_iter=3000, random_state=42),
        'SVM (RBF Kernel)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, max_iter=3000, random_state=42),
        'Random Forest (Tuned)': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'AdaBoost (Ensemble)': AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42),
        'MLP Classifier (Sklearn)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    }
    
    results = []
    trained_models = {}
    
    # 1. Train Scikit-Learn Classifiers
    for name, clf in models.items():
        print(f"--> Training {name}...")
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = clf.predict(X_test)
        
        # ROC AUC
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Algorithm': name,
            'Test Accuracy': round(acc * 100, 2),
            'Precision': round(prec * 100, 2),
            'Recall': round(rec * 100, 2),
            'F1-Score': round(f1 * 100, 2),
            'ROC-AUC': round(auc, 4),
            'Train Time (s)': round(train_time, 2)
        })
        trained_models[name] = clf

    # 2. Train Deep Neural Network (Exp 8B TensorFlow/Keras)
    print("--> Training Deep Neural Network (TensorFlow/Keras)...")
    start_time = time.time()
    
    dnn = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    dnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    dnn.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.15,
        verbose=0
    )
    dnn_train_time = time.time() - start_time
    
    y_prob_dnn = dnn.predict(X_test, verbose=0).flatten()
    y_pred_dnn = (y_prob_dnn >= 0.5).astype(int)
    
    results.append({
        'Algorithm': 'Deep Neural Network (Keras)',
        'Test Accuracy': round(accuracy_score(y_test, y_pred_dnn) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred_dnn, zero_division=0) * 100, 2),
        'Recall': round(recall_score(y_test, y_pred_dnn, zero_division=0) * 100, 2),
        'F1-Score': round(f1_score(y_test, y_pred_dnn, zero_division=0) * 100, 2),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob_dnn), 4),
        'Train Time (s)': round(dnn_train_time, 2)
    })
    trained_models['Deep Neural Network (Keras)'] = dnn
    
    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    return results_df, trained_models

def train_clustering_models(X_scaled, n_clusters=3):
    """
    Train and evaluate K-Means, Agglomerative, and GMM clustering (Exp 10).
    """
    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 2. Agglomerative (sample 3000 to keep memory fast)
    sample_size = min(3000, len(X_scaled))
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X_scaled[:sample_size])
    
    # 3. GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    
    cluster_metrics = [
        {
            'Algorithm': 'K-Means',
            'Silhouette Score': round(silhouette_score(X_scaled[:3000], kmeans_labels[:3000]), 4),
            'Davies-Bouldin': round(davies_bouldin_score(X_scaled[:3000], kmeans_labels[:3000]), 4)
        },
        {
            'Algorithm': 'Agglomerative Clustering',
            'Silhouette Score': round(silhouette_score(X_scaled[:sample_size], agg_labels), 4),
            'Davies-Bouldin': round(davies_bouldin_score(X_scaled[:sample_size], agg_labels), 4)
        },
        {
            'Algorithm': 'Gaussian Mixture Model (GMM)',
            'Silhouette Score': round(silhouette_score(X_scaled[:3000], gmm_labels[:3000]), 4),
            'Davies-Bouldin': round(davies_bouldin_score(X_scaled[:3000], gmm_labels[:3000]), 4)
        }
    ]
    
    return pd.DataFrame(cluster_metrics), kmeans, gmm
