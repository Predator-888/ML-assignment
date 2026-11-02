# Heart Failure Survival Prediction
### Machine Learning Lab Project

A machine learning project that evaluates and compares different classification algorithms to predict the survival outcome of heart failure patients, using the **UCI Heart Failure Clinical Records Dataset** (299 patients, clinical biomarkers).

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Machine Learning Algorithms Tested](#machine-learning-algorithms-tested)
4. [Model Performance & Results](#model-performance--results)
5. [Unsupervised Learning (Clustering)](#unsupervised-learning-clustering)
6. [Treatment Suggestions (Collaborative Filtering)](#treatment-suggestions-collaborative-filtering)
7. [Project Structure](#project-structure)
8. [How to Run the Project](#how-to-run-the-project)
9. [Viva Questions & Answers](#viva-questions--answers)

---

## 🎯 Project Overview
The objective of this project is to apply core machine learning algorithms taught in the ML lab to a real-world healthcare dataset:
1. **Supervised Classification:** Train and test multiple classifiers to predict patient survival (`DEATH_EVENT`: 0 = survived, 1 = deceased).
2. **Model Selection:** Compare all models on the same test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC to pick the best algorithm.
3. **Unsupervised Learning:** Group patients using clustering algorithms (K-Means, Agglomerative, GMM).
4. **Recommendation:** Suggest treatments for patients using collaborative filtering with cosine similarity.
5. **Local Deployment:** Provide an interactive local web interface built with Streamlit.

---

## 📊 Dataset Description
- **Source:** UCI Machine Learning Repository (Davide Chicco & Giuseppe Jurman, 2020)
- **Total Samples:** 299 patient records
- **Features (13 core features + engineered features):**
  - `age`: Patient age
  - `anaemia`: Decrease of red blood cells (0 = No, 1 = Yes)
  - `creatinine_phosphokinase`: Level of CPK enzyme in blood (mcg/L)
  - `diabetes`: If the patient has diabetes (0 = No, 1 = Yes)
  - `ejection_fraction`: Percentage of blood leaving heart at each contraction (%)
  - `high_blood_pressure`: If patient has hypertension (0 = No, 1 = Yes)
  - `platelets`: Platelets in blood (kiloplatelets/mL)
  - `serum_creatinine`: Level of creatinine in blood (mg/dL)
  - `serum_sodium`: Level of sodium in blood (mEq/L)
  - `sex`: 0 = Female, 1 = Male
  - `smoking`: 0 = No, 1 = Yes
  - `time`: Follow-up observation period (days)
  - `DEATH_EVENT`: Target variable (0 = Survived, 1 = Deceased)

---

## 🔬 Machine Learning Algorithms Tested

We evaluated the following algorithms covered in our ML lab coursework:

| Algorithm | Category | Hyperparameters / Details |
|---|---|---|
| **K-Nearest Neighbors (KNN)** | Instance-based | $k=5$, distance weighting |
| **Support Vector Machine (Linear)** | Maximum Margin Classifier | Linear kernel, $C=1.0$ |
| **Support Vector Machine (RBF)** | Kernel Method | RBF kernel, $C=1.0$, $\gamma=\text{scale}$ |
| **Multi-Layer Perceptron (MLP)** | Neural Network | Hidden layers: (32, 16), ReLU activation |
| **Deep Neural Network** | Deep Learning | 4 layers (Dense 32 $\rightarrow$ 16 $\rightarrow$ 1, Dropout 0.2) |
| **Random Forest** | Bagging Ensemble | 100 trees, max depth 6 |
| **AdaBoost** | Boosting Ensemble | 80 decision stumps, learning rate 0.5 |

---

## 📈 Model Performance & Results

All models were evaluated on the same 80/20 train/test split (239 training samples, 60 test samples):

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **🏆 AdaBoost** | **81.67%** | **78.57%** | **57.89%** | **66.67%** | **0.8639** | 0.09s |
| **Random Forest** | 78.33% | 68.75% | 57.89% | 62.86% | 0.8729 | 0.14s |
| **SVM (RBF Kernel)** | 73.33% | 57.89% | 57.89% | 57.89% | 0.8177 | 0.01s |
| **SVM (Linear Kernel)** | 76.67% | 69.23% | 47.37% | 56.25% | 0.8614 | 0.02s |
| **Deep Neural Network** | 71.67% | 55.56% | 52.63% | 54.05% | 0.7715 | 3.64s |
| **KNN (k=5)** | 71.67% | 58.33% | 36.84% | 45.16% | 0.6727 | 0.00s |
| **MLP Classifier** | 31.67% | 31.67% | 100.00% | 48.10% | 0.4082 | 0.02s |

### Best Model Selection:
**AdaBoost** gave the highest **F1-Score (66.67%)** and **Accuracy (81.67%)**.
Because about 68% of patients survived and 32% did not (imbalance), F1-Score and ROC-AUC were used as the primary criteria instead of relying purely on accuracy.

---

## 👥 Unsupervised Learning (Clustering)

We applied clustering to group patients into 3 distinct clinical profiles:

| Algorithm | Silhouette Score | Davies-Bouldin Index |
|---|:---:|:---:|
| **Agglomerative (Hierarchical)** | **0.1316** | **2.2658** |
| K-Means | 0.1056 | 2.4260 |
| Gaussian Mixture Model (GMM) | 0.0834 | 2.7036 |

**Discovered Patient Groups:**
1. **Group 0 (Low Risk):** Preserved ejection fraction (>45%), normal creatinine. Mortality: ~12%.
2. **Group 1 (High Risk):** Reduced ejection fraction (<30%), elevated creatinine (>1.8 mg/dL). Mortality: ~58%.
3. **Group 2 (Elderly with Comorbidities):** Average age > 70 years, multiple chronic conditions. Mortality: ~34%.

---

## 💊 Treatment Suggestions (Collaborative Filtering)

Using collaborative filtering with **Cosine Similarity**:
- **Patient-Based (User-User):** Calculates similarity between patient feature vectors:
  $$\text{Cosine Similarity}(p_1, p_2) = \frac{p_1 \cdot p_2}{\|p_1\| \|p_2\|}$$
  Recommends interventions that patients with similar clinical features responded well to.
- **Intervention-Based (Item-Item):** Identifies treatments frequently prescribed together.

---

## 📁 Project Structure

```
heart_failure_project/
├── data/
│   └── heart_failure_clinical_records.csv # Dataset (299 rows)
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb             # Feature Engineering & Scaling
│   ├── 03_dimensionality_reduction.ipynb  # PCA & t-SNE
│   ├── 04_clustering.ipynb                # K-Means, Agglomerative, GMM
│   ├── 05_model_comparison.ipynb          # Classifier Comparisons
│   ├── 06_clinical_similarity_recommender.ipynb # Collaborative Filtering
│   └── 07_final_report.ipynb              # Final Summary Report
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                   # Data cleaning & scaling pipeline
│   ├── models.py                          # Model definitions & evaluation functions
│   └── recommend.py                       # Collaborative filtering recommender
├── models/
│   ├── best_model.pkl                     # Saved AdaBoost model
│   ├── scaler.pkl                         # Saved StandardScaler
│   ├── feature_names.pkl                  # Feature list
│   ├── cluster_model.pkl                  # Saved K-Means model
│   ├── recommender.pkl                    # Saved recommender
│   └── benchmark_results.csv              # Results table
├── app/
│   └── streamlit_app.py                   # Streamlit web interface
├── train_all.py                           # Training and evaluation script
├── generate_notebooks.py                  # Script to create notebooks
├── requirements.txt                       # Project dependencies
└── README.md                              # Documentation
```

---

## 🚀 How to Run the Project

### 1. Run the Streamlit Web App:
```powershell
streamlit run heart_failure_project/app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

### 2. (Optional) Retrain All Models:
```powershell
python heart_failure_project/train_all.py
```

### 3. Open Notebooks:
Open any notebook in `heart_failure_project/notebooks/` using Jupyter Notebook or VS Code.

---

## 🎓 Viva Questions & Answers

**Q1: Why did you choose AdaBoost over Neural Networks?**
> *Answer:* Neural networks typically require thousands of training examples to generalize well without overfitting. For smaller tabular datasets (299 rows), tree-based ensemble methods like AdaBoost perform better because they build small decision trees sequentially, focusing on errors made by previous trees without overfitting.

**Q2: Why is F1-score more important than Accuracy here?**
> *Answer:* The dataset has a class imbalance (68% survived vs 32% deceased). A dummy model predicting that every patient survives would get 68% accuracy but would fail to identify any patient at risk. F1-score balances precision and recall, ensuring we detect true high-risk cases.

**Q3: What are the most important features for prediction?**
> *Answer:* **Serum Creatinine** and **Ejection Fraction**. Reduced ejection fraction indicates systolic heart failure, while high serum creatinine indicates reduced kidney function, which is closely linked to cardiac complications (cardiorenal syndrome).

**Q4: How does Collaborative Filtering work in this project?**
> *Answer:* We treat patients as users and treatments as items. Using Cosine Similarity on patient clinical profiles, we find the most similar historical patients and recommend the interventions most strongly indicated for that profile.
