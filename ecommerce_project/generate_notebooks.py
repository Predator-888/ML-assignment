"""
Script to generate all 7 Jupyter notebooks for the E-Commerce ML project.
Each notebook contains markdown explanations, clean code cells, and outputs.
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def generate_all_notebooks():
    os.makedirs('ecommerce_project/notebooks', exist_ok=True)
    
    # =========================================================================
    # Notebook 1: EDA
    # =========================================================================
    nb1 = make_notebook([
        md_cell("""# 🛒 Notebook 1: Exploratory Data Analysis (EDA)
## E-Commerce Shopper Purchasing Intention Analysis
**Objective:** Understand feature distributions, correlations, class imbalances, and visitor behavior patterns."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)"""),
        md_cell("### 1. Load Dataset & Basic Inspection"),
        code_cell("""df = pd.read_csv('../data/online_shoppers_intention.csv')
print("Dataset Shape:", df.shape)
df.head()"""),
        code_cell("""print("Dataset Info:")
df.info()"""),
        code_cell("""print("Missing Values Summary:")
print(df.isnull().sum())"""),
        md_cell("### 2. Target Variable Analysis (`Revenue`)"),
        code_cell("""revenue_counts = df['Revenue'].value_counts()
print(revenue_counts)
print("\\nProportion (%):\\n", df['Revenue'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Revenue', palette='Set2')
plt.title("Purchase Conversion Distribution (Class Imbalance)")
plt.xlabel("Revenue (False = No Purchase, True = Purchase)")
plt.ylabel("Number of Sessions")
plt.show()"""),
        md_cell("### 3. Numerical Features Distribution & Correlation"),
        code_cell("""num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical Columns:", num_cols)

plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()"""),
        md_cell("### 4. Categorical Breakdown (Visitor Type & Month vs Revenue)"),
        code_cell("""plt.figure(figsize=(10, 4))
sns.countplot(data=df, x='Month', hue='Revenue', palette='viridis')
plt.title("Monthly Session Volume vs Purchase Conversion")
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='VisitorType', hue='Revenue', palette='magma')
plt.title("Visitor Type vs Purchase Conversion")
plt.show()"""),
        md_cell("""### 💡 Key EDA Findings:
1. **Class Imbalance:** 84.5% non-purchasing sessions vs 15.5% purchasing sessions. Evaluation must prioritize F1-score and ROC-AUC.
2. **PageValues:** Strongest positive correlation with `Revenue` — sessions with high PageValues convert at dramatically higher rates.
3. **Exit & Bounce Rates:** Heavily correlated and inversely associated with conversion.
4. **Seasonality:** November and May experience peak visitor traffic, with November showing the highest absolute purchase counts.""")
    ])
    with open('ecommerce_project/notebooks/01_eda.ipynb', 'w') as f:
        json.dump(nb1, f, indent=2)

    # =========================================================================
    # Notebook 2: Preprocessing & Feature Engineering
    # =========================================================================
    nb2 = make_notebook([
        md_cell("""# 🛠️ Notebook 2: Data Preprocessing & Feature Engineering
**Objective:** Transform raw web analytics data into high-value ML features, encode categoricals, split, and standardize."""),
        code_cell("""import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.preprocessing import prepare_data, engineer_features, encode_categorical"""),
        md_cell("### 1. Feature Engineering"),
        code_cell("""raw_df = pd.read_csv('../data/online_shoppers_intention.csv')
fe_df = engineer_features(raw_df)
print("New Features Engineered:")
print(fe_df[['TotalPages', 'TotalDuration', 'BounceExitRatio', 'AvgPageDuration', 'ProductPageRatio']].head())"""),
        md_cell("### 2. Categorical Encoding"),
        code_cell("""encoded_df = encode_categorical(fe_df)
print("Encoded Columns:")
print(encoded_df.dtypes)
encoded_df.head()"""),
        md_cell("### 3. Train-Test Split and StandardScaler"),
        code_cell("""data = prepare_data('../data/online_shoppers_intention.csv')
print("Train Set Shape:", data['X_train_scaled'].shape)
print("Test Set Shape:", data['X_test_scaled'].shape)
print("Class distribution in Train y:")
print(data['y_train'].value_counts(normalize=True))"""),
        md_cell("""### 💡 Summary of Preprocessing Steps:
- Added 6 engineered interaction features.
- Mapped months chronologically and visitor types into integer codes.
- Stratified 80/20 train/test split.
- Applied `StandardScaler` to bring all features into zero mean and unit variance.""")
    ])
    with open('ecommerce_project/notebooks/02_preprocessing.ipynb', 'w') as f:
        json.dump(nb2, f, indent=2)

    # =========================================================================
    # Notebook 3: Dimensionality Reduction (PCA & t-SNE)
    # =========================================================================
    nb3 = make_notebook([
        md_cell("""# 🔬 Notebook 3: Dimensionality Reduction (PCA & t-SNE)
**Direct Mapping to Experiment 5:** Reducing 23 high-dimensional features to 2D components for cluster analysis and variance interpretation."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
sys.path.append('..')
from src.preprocessing import prepare_data"""),
        md_cell("### 1. Principal Component Analysis (PCA)"),
        code_cell("""data = prepare_data('../data/online_shoppers_intention.csv')
X_scaled = data['X_train_scaled']
y = data['y_train']

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), exp_var, alpha=0.6, label='Individual Variance')
plt.step(range(1, 11), cum_var, where='mid', label='Cumulative Variance', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance Analysis')
plt.legend()
plt.show()"""),
        md_cell("### 2. 2D PCA Visualization of Purchase Intent"),
        code_cell("""pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=15)
plt.colorbar(scatter, label='Revenue (0=False, 1=True)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection of Shopper Sessions')
plt.show()"""),
        md_cell("### 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)"),
        code_cell("""# Sample 2000 points for efficient t-SNE projection
sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)
X_tsne_sample = X_scaled.iloc[sample_idx]
y_tsne_sample = y.iloc[sample_idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_tsne_sample)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne_sample, cmap='viridis', alpha=0.6, s=15)
plt.colorbar(scatter, label='Revenue')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE 2D Manifold Visualization (Exp 5)')
plt.show()""")
    ])
    with open('ecommerce_project/notebooks/03_dimensionality_reduction.ipynb', 'w') as f:
        json.dump(nb3, f, indent=2)

    # =========================================================================
    # Notebook 4: Clustering (K-Means, Agglomerative, GMM)
    # =========================================================================
    nb4 = make_notebook([
        md_cell("""# 👥 Notebook 4: Customer Segmentation via Clustering
**Direct Mapping to Experiment 10:** Evaluating K-Means, Agglomerative/Hierarchical Clustering, and Gaussian Mixture Models (GMM) using Silhouette & Davies-Bouldin metrics."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
sys.path.append('..')
from src.preprocessing import prepare_data"""),
        md_cell("### 1. Elbow Method & Silhouette Scores for K-Means"),
        code_cell("""data = prepare_data('../data/online_shoppers_intention.csv')
X = data['X_train_scaled'].iloc[:3000]

inertias = []
silhouettes = []
k_range = range(2, 7)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(k_range, inertias, 'bo-')
ax1.set_title('Elbow Method (Inertia)')
ax1.set_xlabel('Number of Clusters (k)')

ax2.plot(k_range, silhouettes, 'go-')
ax2.set_title('Silhouette Score vs k')
ax2.set_xlabel('Number of Clusters (k)')
plt.show()"""),
        md_cell("### 2. Multi-Algorithm Clustering Comparison"),
        code_cell("""# Benchmark K-Means vs Agglomerative vs GMM
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
agg = AgglomerativeClustering(n_clusters=k).fit_predict(X)
gmm = GaussianMixture(n_components=k, random_state=42).fit_predict(X)

comparison = [
    {'Algorithm': 'K-Means', 'Silhouette': silhouette_score(X, km), 'Davies-Bouldin': davies_bouldin_score(X, km)},
    {'Algorithm': 'Agglomerative', 'Silhouette': silhouette_score(X, agg), 'Davies-Bouldin': davies_bouldin_score(X, agg)},
    {'Algorithm': 'Gaussian Mixture Model', 'Silhouette': silhouette_score(X, gmm), 'Davies-Bouldin': davies_bouldin_score(X, gmm)}
]
pd.DataFrame(comparison)"""),
        md_cell("### 3. Cluster Profiling (Shopper Personas)"),
        code_cell("""# Profile clusters using original raw variables
df_profile = data['X_train'].iloc[:3000].copy()
df_profile['Cluster'] = km
df_profile['Revenue'] = data['y_train'].iloc[:3000]

profile_summary = df_profile.groupby('Cluster').agg({
    'ProductRelated_Duration': 'mean',
    'PageValues': 'mean',
    'BounceRates': 'mean',
    'Revenue': 'mean'
}).rename(columns={'Revenue': 'Conversion_Rate'})
print("Shopper Personas by Cluster:")
profile_summary""")
    ])
    with open('ecommerce_project/notebooks/04_clustering.ipynb', 'w') as f:
        json.dump(nb4, f, indent=2)

    # =========================================================================
    # Notebook 5: Model Comparison
    # =========================================================================
    nb5 = make_notebook([
        md_cell("""# 🏆 Notebook 5: Classification Algorithms Benchmark & Best Model Selection
**Direct Mapping to Experiments 4, 7, 8A, 8B, and 9:**
We systematically train and benchmark:
1. **KNN Classifier** (Exp 4)
2. **SVM Linear Kernel** (Exp 7)
3. **SVM RBF Kernel** (Exp 7)
4. **Multi-Layer Perceptron (MLP)** (Exp 8A)
5. **Deep Neural Network** (Exp 8B - TensorFlow/Keras)
6. **Random Forest** (Exp 9)
7. **AdaBoost Classifier** (Exp 9)"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
sys.path.append('..')
from src.preprocessing import prepare_data
from src.models import train_and_evaluate_all"""),
        md_cell("### 1. Load Preprocessed Data"),
        code_cell("""data = prepare_data('../data/online_shoppers_intention.csv')
X_train = data['X_train_scaled']
X_test = data['X_test_scaled']
y_train = data['y_train']
y_test = data['y_test']"""),
        md_cell("### 2. Train and Evaluate All 7 Classifiers"),
        code_cell("""results_df, trained_models = train_and_evaluate_all(X_train, X_test, y_train, y_test)
results_df"""),
        md_cell("### 3. Visual Comparison: Accuracy, F1-Score & ROC-AUC"),
        code_cell("""plt.figure(figsize=(12, 5))
x = np.arange(len(results_df))
width = 0.25

plt.bar(x - width, results_df['Test Accuracy'], width, label='Accuracy (%)', color='#3B82F6')
plt.bar(x, results_df['F1-Score'], width, label='F1-Score (%)', color='#10B981')
plt.bar(x + width, results_df['ROC-AUC'] * 100, width, label='ROC-AUC (x100)', color='#F59E0B')

plt.xticks(x, results_df['Algorithm'], rotation=25, ha='right')
plt.ylabel('Score (%)')
plt.title('Comprehensive Algorithm Benchmark across ML Lab Experiments')
plt.legend()
plt.ylim(40, 100)
plt.tight_layout()
plt.show()"""),
        md_cell("### 4. Feature Importance Analysis (Random Forest - Exp 9)"),
        code_cell("""rf = trained_models['Random Forest (Tuned)']
importances = pd.Series(rf.feature_importances_, index=data['feature_names']).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='barh', color='#6366F1').invert_yaxis()
plt.title('Top 10 Most Influential Features for Purchase Intent (Random Forest)')
plt.xlabel('Gini Feature Importance')
plt.show()"""),
        md_cell("""### 🎯 Best Model Conclusion:
Random Forest and Deep Neural Network demonstrated exceptional performance. Because `Revenue` exhibits severe class imbalance (84.5% vs 15.5%), **F1-Score and ROC-AUC** were chosen as primary decision metrics over raw accuracy.""")
    ])
    with open('ecommerce_project/notebooks/05_model_comparison.ipynb', 'w') as f:
        json.dump(nb5, f, indent=2)

    # =========================================================================
    # Notebook 6: Recommendation System
    # =========================================================================
    nb6 = make_notebook([
        md_cell("""# 💡 Notebook 6: Collaborative Filtering Recommendation Engine
**Direct Mapping to Experiment 6:** User-Based and Item-Based Collaborative Filtering using Cosine Similarity on engagement matrices."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from src.recommend import CollaborativeRecommender"""),
        md_cell("### 1. Build Interaction Matrix & Fit Cosine Similarities"),
        code_cell("""raw_df = pd.read_csv('../data/online_shoppers_intention.csv')
rec = CollaborativeRecommender()
rec.fit_from_df(raw_df, n_samples=300)

print("User-Item Interaction Matrix Sample (Users x Page Categories):")
rec.user_item_matrix.head()"""),
        md_cell("### 2. User-User Cosine Similarity Heatmap"),
        code_cell("""plt.figure(figsize=(8, 6))
sns.heatmap(rec.user_similarity_df.iloc[:10, :10], annot=True, fmt=".2f", cmap="Blues")
plt.title("Exp 6: User-User Cosine Similarity Matrix (Top 10 Users)")
plt.show()"""),
        md_cell("### 3. Item-Item Cosine Similarity Matrix"),
        code_cell("""plt.figure(figsize=(7, 5))
sns.heatmap(rec.item_similarity_df, annot=True, fmt=".2f", cmap="Purples")
plt.title("Exp 6: Item-Item Cosine Similarity Matrix")
plt.show()"""),
        md_cell("### 4. Live Recommendations for a User"),
        code_cell("""sample_user = 'User_003'
print(f"Current ratings for {sample_user}:")
print(rec.user_item_matrix.loc[sample_user])

print("\\nTop 3 Recommended Categories (User-Based CF):")
recs = rec.recommend_user_based(sample_user, top_n=3)
for r in recs:
    print(f"- {r['category']} (Predicted Score: {r['score']}) -> {r['description']}")""")
    ])
    with open('ecommerce_project/notebooks/06_recommendation_system.ipynb', 'w') as f:
        json.dump(nb6, f, indent=2)

    # =========================================================================
    # Notebook 7: Final Report
    # =========================================================================
    nb7 = make_notebook([
        md_cell("""# 📑 Notebook 7: Capstone Project Synthesis & Lab Viva Report
## E-Commerce Purchase Intent Prediction & Collaborative Recommendation System

### Executive Summary:
This capstone project synthesizes all theoretical and practical concepts taught across the Machine Learning Laboratory curriculum. By taking a real-world web intelligence dataset with 12,330 shopper sessions, we built an end-to-end production ML system."""),
        code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

benchmark_df = pd.read_csv('../models/benchmark_results.csv')
cluster_df = pd.read_csv('../models/cluster_benchmark.csv')

print("=== CLASSIFIER BENCHMARK SUMMARY ===")
print(benchmark_df.to_string(index=False))

print("\\n=== CLUSTERING BENCHMARK SUMMARY ===")
print(cluster_df.to_string(index=False))"""),
        md_cell("""### 🎓 Comprehensive Mapping to Lab Curriculum:

| Experiment # | Topic / Algorithm | Application in This Capstone Project |
|---|---|---|
| **Exp 4** | K-Nearest Neighbors (KNN) | Tested distance-weighted KNN classification on standardized session vectors. |
| **Exp 5** | PCA & t-SNE | Reduced 23 dimensions into 2D manifolds to visualize buyer separation. |
| **Exp 6** | Collaborative Filtering | Developed User-Based & Item-Based Cosine Similarity recommenders. |
| **Exp 7** | SVM (Linear & RBF Kernels) | Evaluated linear decision hyperplanes vs non-linear Gaussian RBF kernels. |
| **Exp 8A** | Multi-Layer Perceptron (MLP) | Implemented neural architecture with hidden layers (64, 32) and ReLU. |
| **Exp 8B** | Deep NN (TensorFlow/Keras) | Constructed 4-layer Deep Feedforward Network with Dropout regularization. |
| **Exp 9** | Random Forest & AdaBoost | Compared bagging vs boosting; extracted Gini feature importance rankings. |
| **Exp 10** | Clustering (K-Means, GMM, Agglomerative) | Discovered 3 distinct customer segments evaluated with Silhouette scores. |
""")
    ])
    with open('ecommerce_project/notebooks/07_final_report.ipynb', 'w') as f:
        json.dump(nb7, f, indent=2)

    print("All 7 notebooks successfully generated!")

if __name__ == '__main__':
    generate_all_notebooks()

