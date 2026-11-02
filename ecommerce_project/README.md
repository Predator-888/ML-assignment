# 🛒 E-Commerce Purchase Intent Prediction & Recommendation Studio
### Machine Learning Laboratory Capstone Project

An end-to-end Machine Learning system that unifies and benchmarks every core algorithm taught in the ML curriculum (Experiments 4 to 10) on a real-world web intelligence dataset with **12,330 shopper sessions**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [ML Lab Experiment Mapping](#ml-lab-experiment-mapping)
3. [Algorithm Benchmark Results](#algorithm-benchmark-results)
4. [Customer Segmentation (Clustering)](#customer-segmentation-clustering)
5. [Recommendation Engine (Collaborative Filtering)](#recommendation-engine-collaborative-filtering)
6. [Project Architecture & File Structure](#project-architecture--file-structure)
7. [How to Run Locally](#how-to-run-locally)
8. [Viva Q&A Preparation](#viva-qa-preparation)

---

## 🎯 Project Overview
In modern e-commerce, identifying high-intent shoppers in real-time allows platforms to offer dynamic incentives, reduce cart abandonment, and personalize navigation. This project solves three key challenges:
1. **Supervised Classification:** Predict whether an active user session will result in a purchase (`Revenue = TRUE/FALSE`).
2. **Unsupervised Customer Segmentation:** Cluster shoppers into behavioral personas to guide marketing strategies.
3. **Collaborative Recommendation Engine:** Recommend the next most relevant site categories or product showcases based on user engagement similarity.

---

## 🎓 ML Lab Experiment Mapping

Every experiment from your lab curriculum has been directly implemented, evaluated, and unified into this project:

| Experiment | Lab Topic / Algorithm | Project Implementation |
|---|---|---|
| **Exp 4** | **K-Nearest Neighbors (KNN)** | Tested distance-weighted KNN ($k=7$) on scaled feature vectors. |
| **Exp 5** | **Dimensionality Reduction (PCA & t-SNE)** | Analyzed explained variance with PCA and visualized 2D manifolds with t-SNE. |
| **Exp 6** | **Collaborative Filtering** | Built User-Based and Item-Based recommendation engines using Cosine Similarity. |
| **Exp 7** | **Support Vector Machine (Linear & RBF)** | Compared linear decision hyperplanes vs non-linear Gaussian RBF kernels. |
| **Exp 8A** | **Multi-Layer Perceptron (MLP)** | Implemented fully-connected neural network with ReLU & early stopping. |
| **Exp 8B** | **Deep Neural Network (TensorFlow/Keras)** | Built 4-layer Deep Feedforward Network with Dropout regularization. |
| **Exp 9** | **Ensemble Methods (RF & AdaBoost)** | Compared Bagging (Random Forest) vs Boosting (AdaBoost); extracted Gini importances. |
| **Exp 10** | **Clustering (K-Means, GMM, Agglomerative)** | Discovered 3 natural customer segments evaluated via Silhouette & Davies-Bouldin scores. |

---

## 📊 Algorithm Benchmark Results

All 7 classification algorithms were trained and evaluated on the exact same stratified 80/20 train/test split (9,864 train / 2,466 test):

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **🏆 AdaBoost (Ensemble)** | **89.09%** | **66.67%** | **59.16%** | **62.69%** | **0.9111** | 0.92s |
| **Random Forest (Tuned)** | 89.98% | 75.09% | 52.88% | 62.06% | 0.9123 | 0.28s |
| **Deep Neural Network (Keras)** | 89.17% | 69.10% | 54.45% | 60.91% | 0.9010 | 3.97s |
| **MLP Classifier (Sklearn)** | 88.89% | 68.12% | 53.14% | 59.71% | 0.8954 | 0.91s |
| **SVM (RBF Kernel)** | 88.32% | 70.80% | 41.88% | 52.63% | 0.8544 | 5.77s |
| **KNN (k=7)** | 87.55% | 69.84% | 34.55% | 46.23% | 0.7823 | 0.00s |
| **SVM (Linear Kernel)** | 33.33% | 15.41% | 73.56% | 25.48% | 0.5346 | 8.55s |

> **Key Viva Insight:** Since the dataset exhibits an **84.5% vs 15.5% class imbalance**, Accuracy is misleading (a dummy classifier guessing 'No Purchase' gets 84.5% accuracy). **F1-Score and ROC-AUC** are the true metrics of model quality. **AdaBoost** and **Random Forest** achieved the best balance of precision and recall.

---

## 👥 Customer Segmentation (Clustering)

Using unsupervised learning algorithms from **Exp 10**, we evaluated 3 clustering techniques ($k=3$):

| Clustering Algorithm | Silhouette Score | Davies-Bouldin Index |
|---|:---:|:---:|
| **K-Means** | **0.2886** | **1.5415** |
| Agglomerative (Hierarchical) | 0.1914 | 1.8656 |
| Gaussian Mixture Model (GMM) | 0.0400 | 3.2307 |

### 🛍️ Discovered Customer Personas:
1. **Cluster 0: High-Value Buyers (~68% conversion)**
   - High time on site, high `PageValues`, focused product browsing.
   - *Action:* Fast checkout and premium loyalty rewards.
2. **Cluster 1: Window Shoppers (~6% conversion)**
   - High bounce and exit rates, low page values.
   - *Action:* Exit-intent discounts, flash sales.
3. **Cluster 2: Account & Support Returning Visitors (~18% conversion)**
   - High administrative pages, returning visitors.
   - *Action:* Order tracking shortcuts and cross-sell recommendations.

---

## 💡 Recommendation Engine (Collaborative Filtering)

Directly modeling **Exp 6**:
- **User-Based Collaborative Filtering:** Computes Cosine Similarity across shopper engagement vectors. If User A is similar to User B, unvisited categories that User B spent significant time on are recommended to User A.
- **Item-Based Collaborative Filtering:** Computes Cosine Similarity across site categories based on co-engagement. When a user explores `ProductRelated`, the system recommends `Special_Offers` and `Cart_Checkout`.

---

## 📁 Project Structure

```
ecommerce_project/
├── data/
│   └── online_shoppers_intention.csv     # 12,330 sessions x 18 attributes
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb            # Engineering & Scaling
│   ├── 03_dimensionality_reduction.ipynb # Exp 5: PCA & t-SNE
│   ├── 04_clustering.ipynb               # Exp 10: K-Means, Agglomerative, GMM
│   ├── 05_model_comparison.ipynb         # Exp 4, 7, 8A, 8B, 9: Classifiers
│   ├── 06_recommendation_system.ipynb    # Exp 6: Collaborative Filtering
│   └── 07_final_report.ipynb             # Synthesis Report & Viva summary
├── src/
│   ├── preprocessing.py                  # Data pipeline & feature engineering
│   ├── models.py                         # Training & evaluation routines
│   └── recommend.py                      # User & Item collaborative filtering
├── models/
│   ├── best_model.pkl                    # Serialized champion model (AdaBoost)
│   ├── scaler.pkl                        # Fitted StandardScaler
│   ├── feature_names.pkl                 # Feature list
│   ├── cluster_model.pkl                 # K-Means model
│   ├── recommender.pkl                   # Fitted recommender
│   └── benchmark_results.csv             # Comparative metrics table
├── app/
│   └── streamlit_app.py                  # Local interactive web app
├── train_all.py                          # Master execution script
├── requirements.txt                      # Dependencies
└── README.md                             # Project Documentation
```

---

## 🚀 How to Run Locally

### 1. Launch the Interactive Web App:
Open PowerShell or your terminal in the project directory:
```bash
streamlit run ecommerce_project/app/streamlit_app.py
```
This opens the web browser at `http://localhost:8501` featuring:
- 🔮 **Purchase Intent Predictor** with live input sliders and probability gauge.
- 👥 **Customer Segmentation Studio** with cluster personas.
- 💡 **Collaborative Recommendation Engine** (User-based & Item-based).
- 📊 **Model Comparison Dashboard** with interactive performance charts and syllabus mapping.

### 2. (Optional) Retrain All Models:
```bash
python ecommerce_project/train_all.py
```

### 3. Open Any Notebook:
Open VS Code or Jupyter Lab and browse `ecommerce_project/notebooks/`.

---

## 🎓 Viva Q&A Preparation

**Q1: Why did you choose F1-score instead of Accuracy for model selection?**
> *Answer:* The dataset has an 84.5% vs 15.5% class imbalance. A naive model that always predicts 'False' would achieve 84.5% accuracy but zero utility. F1-score balances Precision (avoiding false alarms) and Recall (catching actual buyers), making it the appropriate metric.

**Q2: Why did AdaBoost and Random Forest outperform Linear SVM?**
> *Answer:* Web session data contains non-linear interactions between duration, bounce rates, and page values. Linear SVM assumes a planar decision boundary, whereas tree-based ensembles create orthogonal axis-aligned non-linear partitions and handle feature scaling and outliers effectively.

**Q3: How does your Collaborative Filtering recommendation work?**
> *Answer:* We built an interaction matrix where rows represent sessions and columns represent site categories. Using Cosine Similarity:
> $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
> In user-based CF, ratings are predicted via weighted averages of top-$k$ nearest neighbors. In item-based CF, items are recommended based on similarity to items the user already interacted with.

**Q4: How does PCA differ from t-SNE in your dimensionality reduction?**
> *Answer:* PCA is a linear technique that maximizes global variance along orthogonal eigenvectors. t-SNE is a non-linear probabilistic technique that minimizes divergence between low-dimensional and high-dimensional probability distributions, preserving local neighborhood structures.

