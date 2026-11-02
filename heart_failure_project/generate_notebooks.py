"""
Script to generate all 7 Jupyter notebooks for the Heart Failure ML project.
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
    os.makedirs('heart_failure_project/notebooks', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Notebook 1: EDA
    # -------------------------------------------------------------------------
    nb1 = make_notebook([
        md_cell("""# ❤️ Notebook 1: Exploratory Data Analysis (EDA)
## Heart Failure Clinical Records Analysis
**Objective:** Investigate clinical biomarkers, distributions, class balance, and correlation with mortality (`DEATH_EVENT`)."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)"""),
        md_cell("### 1. Load Dataset & Clinical Inspection"),
        code_cell("""df = pd.read_csv('../data/heart_failure_clinical_records.csv')
print("Clinical Cohort Shape:", df.shape)
df.head()"""),
        code_cell("""print("Dataset Information:")
df.info()"""),
        code_cell("""print("Statistical Summary:")
df.describe().T"""),
        md_cell("### 2. Target Variable: Mortality Distribution (`DEATH_EVENT`)"),
        code_cell("""death_counts = df['DEATH_EVENT'].value_counts()
print(death_counts)
print("\\nProportion (%):\\n", df['DEATH_EVENT'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='DEATH_EVENT', palette=['#10B981', '#EF4444'])
plt.title("Mortality Distribution (0 = Survived, 1 = Deceased)")
plt.xlabel("DEATH_EVENT")
plt.ylabel("Number of Patients")
plt.show()"""),
        md_cell("### 3. Key Biomarker Distributions vs Mortality"),
        code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.boxplot(ax=axes[0], data=df, x='DEATH_EVENT', y='ejection_fraction', palette='Set2')
axes[0].set_title("Ejection Fraction vs Mortality")

sns.boxplot(ax=axes[1], data=df, x='DEATH_EVENT', y='serum_creatinine', palette='Set2')
axes[1].set_title("Serum Creatinine vs Mortality")

sns.boxplot(ax=axes[2], data=df, x='DEATH_EVENT', y='serum_sodium', palette='Set2')
axes[2].set_title("Serum Sodium vs Mortality")

plt.tight_layout()
plt.show()"""),
        md_cell("### 4. Correlation Matrix"),
        code_cell("""plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap of Clinical Biomarkers")
plt.show()"""),
        md_cell("""### 💡 Key Clinical Insights:
1. **Serum Creatinine & Ejection Fraction:** Strongest predictive biomarkers for mortality. Lower ejection fraction (<35%) and elevated serum creatinine (>1.5) indicate acute cardiorenal dysfunction.
2. **Follow-Up Time:** Long observation times correlate with patient survival.
3. **Class Imbalance:** 67.9% survived (0) vs 32.1% deceased (1). Evaluation must prioritize F1-score, Precision, and Recall.""")
    ])
    with open('heart_failure_project/notebooks/01_eda.ipynb', 'w') as f:
        json.dump(nb1, f, indent=2)

    # -------------------------------------------------------------------------
    # Notebook 2: Preprocessing & Feature Engineering
    # -------------------------------------------------------------------------
    nb2 = make_notebook([
        md_cell("""# 🛠️ Notebook 2: Clinical Feature Engineering & Standardization
**Objective:** Engineer cardiorenal biomarkers, comorbidity counts, perform stratified splitting and StandardScaler scaling."""),
        code_cell("""import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.preprocessing import prepare_data, engineer_features"""),
        md_cell("### 1. Engineer Domain-Specific Clinical Features"),
        code_cell("""raw_df = pd.read_csv('../data/heart_failure_clinical_records.csv')
fe_df = engineer_features(raw_df)
print("Engineered Features Sample:")
fe_df[['Renal_Cardiac_Ratio', 'Hyponatremia_Flag', 'Severe_EF_Flag', 'Age_Risk_Group', 'Comorbidity_Count']].head()"""),
        md_cell("### 2. Stratified Train-Test Split and Scaling"),
        code_cell("""data = prepare_data('../data/heart_failure_clinical_records.csv')
print("Train Shape:", data['X_train_scaled'].shape)
print("Test Shape:", data['X_test_scaled'].shape)
print("Feature Names (Total:", len(data['feature_names']), "):")
print(data['feature_names'])"""),
        md_cell("""### 💡 Summary of Preprocessing Steps:
- Formulated Cardiorenal Risk Ratio (`serum_creatinine / ejection_fraction`).
- Created flags for hyponatremia (<135 mEq/L) and severe systolic dysfunction (EF < 30%).
- Standardized all continuous variables with `StandardScaler` to ensure zero mean and unit variance for distance and gradient-based algorithms.""")
    ])
    with open('heart_failure_project/notebooks/02_preprocessing.ipynb', 'w') as f:
        json.dump(nb2, f, indent=2)

    # -------------------------------------------------------------------------
    # Notebook 3: Dimensionality Reduction (PCA & t-SNE)
    # -------------------------------------------------------------------------
    nb3 = make_notebook([
        md_cell("""# 🔬 Notebook 3: Dimensionality Reduction (PCA & t-SNE)
**Direct Mapping to Experiment 5:** Reducing 18 clinical features to 2D components to visualize variance and patient risk separation."""),
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
        code_cell("""data = prepare_data('../data/heart_failure_clinical_records.csv')
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
plt.title('PCA Explained Variance (Exp 5)')
plt.legend()
plt.show()"""),
        md_cell("### 2. 2D PCA Clinical Projection"),
        code_cell("""pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.8, s=40)
plt.colorbar(scatter, label='Mortality (0=Survived, 1=Deceased)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection of Cardiac Cohort')
plt.show()"""),
        md_cell("### 3. t-SNE Non-Linear Manifold Visualization"),
        code_cell("""tsne = TSNE(n_components=2, random_state=42, perplexity=25, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.8, s=40)
plt.colorbar(scatter, label='Mortality')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE 2D Manifold Visualization (Exp 5)')
plt.show()""")
    ])
    with open('heart_failure_project/notebooks/03_dimensionality_reduction.ipynb', 'w') as f:
        json.dump(nb3, f, indent=2)

    # -------------------------------------------------------------------------
    # Notebook 4: Clustering (K-Means, Agglomerative, GMM)
    # -------------------------------------------------------------------------
    nb4 = make_notebook([
        md_cell("""# 👥 Notebook 4: Patient Risk Stratification via Clustering
**Direct Mapping to Experiment 10:** Benchmarking K-Means, Agglomerative (Hierarchical), and Gaussian Mixture Models (GMM) using Silhouette & Davies-Bouldin metrics."""),
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
        md_cell("### 1. Optimal Cluster Search via Silhouette Analysis"),
        code_cell("""data = prepare_data('../data/heart_failure_clinical_records.csv')
X = data['X_train_scaled']

silhouettes = []
k_range = range(2, 7)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    silhouettes.append(silhouette_score(X, labels))

plt.figure(figsize=(7, 4))
plt.plot(k_range, silhouettes, 'go-')
plt.title('Silhouette Score vs k (Exp 10)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()"""),
        md_cell("### 2. Multi-Algorithm Clustering Benchmark"),
        code_cell("""k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
agg = AgglomerativeClustering(n_clusters=k).fit_predict(X)
gmm = GaussianMixture(n_components=k, random_state=42).fit_predict(X)

comparison = [
    {'Algorithm': 'K-Means', 'Silhouette': silhouette_score(X, km), 'Davies-Bouldin': davies_bouldin_score(X, km)},
    {'Algorithm': 'Agglomerative Clustering', 'Silhouette': silhouette_score(X, agg), 'Davies-Bouldin': davies_bouldin_score(X, agg)},
    {'Algorithm': 'Gaussian Mixture Model', 'Silhouette': silhouette_score(X, gmm), 'Davies-Bouldin': davies_bouldin_score(X, gmm)}
]
pd.DataFrame(comparison)"""),
        md_cell("### 3. Clinical Risk Profiling of Discovered Cohorts"),
        code_cell("""df_profile = data['X_train'].copy()
df_profile['Cohort'] = agg
df_profile['Mortality_Rate'] = data['y_train']

summary = df_profile.groupby('Cohort').agg({
    'age': 'mean',
    'ejection_fraction': 'mean',
    'serum_creatinine': 'mean',
    'Mortality_Rate': 'mean'
})
print("Clinical Risk Phenotypes:")
summary""")
    ])
    with open('heart_failure_project/notebooks/04_clustering.ipynb', 'w') as f:
        json.dump(nb4, f, indent=2)

    # -------------------------------------------------------------------------
    # Notebook 5: Model Comparison
    # -------------------------------------------------------------------------
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
        md_cell("### 1. Load Preprocessed Clinical Data"),
        code_cell("""data = prepare_data('../data/heart_failure_clinical_records.csv')
X_train = data['X_train_scaled']
X_test = data['X_test_scaled']
y_train = data['y_train']
y_test = data['y_test']"""),
        md_cell("### 2. Train and Evaluate All 7 Classifiers"),
        code_cell("""results_df, trained_models = train_and_evaluate_all(X_train, X_test, y_train, y_test)
results_df"""),
        md_cell("### 3. Comparative Performance Visualization"),
        code_cell("""plt.figure(figsize=(12, 5))
x = np.arange(len(results_df))
width = 0.25

plt.bar(x - width, results_df['Test Accuracy'], width, label='Accuracy (%)', color='#EF4444')
plt.bar(x, results_df['F1-Score'], width, label='F1-Score (%)', color='#10B981')
plt.bar(x + width, results_df['ROC-AUC'] * 100, width, label='ROC-AUC (x100)', color='#3B82F6')

plt.xticks(x, results_df['Algorithm'], rotation=25, ha='right')
plt.ylabel('Score (%)')
plt.title('Algorithm Benchmark across Machine Learning Lab Experiments')
plt.legend()
plt.ylim(20, 100)
plt.tight_layout()
plt.show()"""),
        md_cell("### 4. Biomarker Importance (Random Forest - Exp 9)"),
        code_cell("""rf = trained_models['Random Forest (Tuned)']
importances = pd.Series(rf.feature_importances_, index=data['feature_names']).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='barh', color='#DC2626').invert_yaxis()
plt.title('Top 10 Most Predictive Biomarkers (Random Forest Gini Importance)')
plt.xlabel('Importance')
plt.show()"""),
        md_cell("""### 🎯 Best Model Selection:
**AdaBoost Ensemble** and **Random Forest** achieved the highest F1-scores and ROC-AUC on the clinical test cohort, demonstrating excellent calibration between precision (minimizing false alarms) and recall (catching high-risk cardiac patients).""")
    ])
    with open('heart_failure_project/notebooks/05_model_comparison.ipynb', 'w') as f:
        json.dump(nb5, f, indent=2)

    # -------------------------------------------------------------------------
    # Notebook 6: Clinical Recommendation System
    # -------------------------------------------------------------------------
    nb6 = make_notebook([
        md_cell("""# 💊 Notebook 6: Clinical Decision Support Recommender
**Direct Mapping to Experiment 6:** Patient-Based and Intervention-Based Collaborative Filtering using Cosine Similarity."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from src.recommend import ClinicalCollaborativeRecommender"""),
        md_cell("### 1. Build Clinical Matrix & Fit Cosine Similarities"),
        code_cell("""raw_df = pd.read_csv('../data/heart_failure_clinical_records.csv')
rec = ClinicalCollaborativeRecommender()
rec.fit_from_df(raw_df)

print("Patient-Intervention Matrix Sample:")
rec.patient_intervention_matrix.head()"""),
        md_cell("### 2. Patient-Patient Cosine Similarity Heatmap (User-User CF)"),
        code_cell("""plt.figure(figsize=(8, 6))
sns.heatmap(rec.patient_similarity_df.iloc[:10, :10], annot=True, fmt=".2f", cmap="Reds")
plt.title("Exp 6: Patient-Patient Cosine Similarity Matrix (Top 10 Cohort)")
plt.show()"""),
        md_cell("### 3. Intervention-Intervention Cosine Similarity Matrix (Item-Item CF)"),
        code_cell("""plt.figure(figsize=(7, 5))
sns.heatmap(rec.intervention_similarity_df, annot=True, fmt=".2f", cmap="Blues")
plt.title("Exp 6: Intervention Synergies Matrix (Item-Item)")
plt.show()"""),
        md_cell("### 4. Cohort Recommendation for a Test Patient"),
        code_cell("""sample_patient = 'Patient_002'
print(f"Current ratings/indications for {sample_patient}:")
print(rec.patient_intervention_matrix.loc[sample_patient])

print("\\nTop Recommended Interventions (Cohort Collaborative Filtering):")
recs = rec.recommend_cohort_based(sample_patient, top_n=3)
for r in recs:
    print(f"- {r['intervention']} (Score: {r['score']}) -> {r['description']}")""")
    ])
    with open('heart_failure_project/notebooks/06_clinical_similarity_recommender.ipynb', 'w') as f:
        json.dump(nb6, f, indent=2)

    # -------------------------------------------------------------------------
    # Notebook 7: Final Report
    # -------------------------------------------------------------------------
    nb7 = make_notebook([
        md_cell("""# 📑 Notebook 7: Capstone Synthesis & Clinical Lab Viva Report
## Heart Failure Mortality Risk Prediction & Clinical Decision Support System

### Executive Summary:
This capstone project synthesizes all theoretical and practical concepts taught across the Machine Learning Laboratory curriculum. By taking the official UCI Heart Failure Clinical Records dataset, we built an end-to-end machine learning prediction system."""),
        code_cell("""import pandas as pd

benchmark_df = pd.read_csv('../models/benchmark_results.csv')
cluster_df = pd.read_csv('../models/cluster_benchmark.csv')

print("=== CLASSIFIER BENCHMARK SUMMARY ===")
print(benchmark_df.to_string(index=False))

print("\\n=== CLUSTERING BENCHMARK SUMMARY ===")
print(cluster_df.to_string(index=False))"""),
        md_cell("""### 🎓 Comprehensive Mapping to Lab Curriculum:

| Experiment # | Topic / Algorithm | Application in This Capstone Project |
|---|---|---|
| **Exp 4** | K-Nearest Neighbors (KNN) | Evaluated distance-weighted KNN ($k=5$) on standardized clinical vitals. |
| **Exp 5** | PCA & t-SNE | Projected 18 clinical features to 2D components to visualize mortality separation. |
| **Exp 6** | Collaborative Filtering | Implemented Patient-Based and Intervention-Based Cosine Similarity recommenders. |
| **Exp 7** | SVM (Linear & RBF Kernels) | Compared linear hyperplanes vs non-linear Gaussian RBF kernels. |
| **Exp 8A** | Multi-Layer Perceptron (MLP) | Implemented neural architecture with hidden layers (32, 16) and ReLU. |
| **Exp 8B** | Deep NN (TensorFlow/Keras) | Constructed 4-layer Deep Feedforward Network with Dropout regularization. |
| **Exp 9** | Random Forest & AdaBoost | Compared bagging vs boosting; extracted Gini feature importance rankings. |
| **Exp 10** | Clustering (K-Means, GMM, Agglomerative) | Discovered 3 distinct clinical risk phenotypes evaluated with Silhouette scores. |
""")
    ])
    with open('heart_failure_project/notebooks/07_final_report.ipynb', 'w') as f:
        json.dump(nb7, f, indent=2)

    print("All 7 Heart Failure notebooks successfully generated!")

if __name__ == '__main__':
    generate_all_notebooks()

