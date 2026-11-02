# 🎓 Heart Failure Survival Prediction — Faculty Presentation & Viva Guide
### Machine Learning Laboratory Capstone Project

> **Note for the Presenter:** This document contains all the technical details, algorithm comparisons, mathematical formulas, experimental results, and viva Q&A needed to present and defend this project to the faculty.

---

## 📌 1. Project Overview & Abstract

- **Project Title:** Heart Failure Patient Survival Prediction and Clinical Decision Support Using Machine Learning
- **Core Objective:** To benchmark multiple supervised and unsupervised machine learning algorithms on real-world patient clinical records, select the best-performing model based on balanced evaluation metrics (F1-Score and ROC-AUC), and deploy a functional decision support interface for clinicians.
- **Dataset:** UCI Heart Failure Clinical Records Dataset (299 patient records, 13 clinical vitals and laboratory biomarkers).
- **Target Variable:** `DEATH_EVENT` (Binary classification: `0` = Patient survived, `1` = Patient deceased during follow-up).
- **Best Model Selected:** **AdaBoost Classifier** (81.67% Test Accuracy, 66.67% F1-Score, 0.8639 ROC-AUC).

---

## 🔬 2. Complete Mapping to Lab Syllabus (Experiments 4–10)

This project directly implements and unifies all algorithms covered in the lab curriculum:

| Experiment | Algorithm / Topic | Role in This Project | Key Hyperparameters / Details |
|---|---|---|---|
| **Exp 4** | **K-Nearest Neighbors (KNN)** | Supervised Classification | $k=5$, distance-weighted voting (`weights='distance'`). Achieved 71.67% accuracy. |
| **Exp 5** | **Dimensionality Reduction (PCA & t-SNE)** | Feature Space Analysis & Visualization | Applied PCA (analyzed explained variance ratio) and non-linear t-SNE to project 18 dimensions to 2D components. |
| **Exp 6** | **Collaborative Filtering** | Treatment Recommendation Engine | Implemented User-Based (patient-patient) and Item-Based (treatment-treatment) collaborative filtering using **Cosine Similarity**. |
| **Exp 7** | **Support Vector Machine (SVM)** | Supervised Classification | Compared Linear Kernel ($C=1.0$) vs Non-Linear Gaussian RBF Kernel ($C=1.0, \gamma=\text{scale}$). RBF achieved 57.89% F1-score. |
| **Exp 8A** | **Multi-Layer Perceptron (MLP)** | Neural Network Classification | Fully connected network with hidden layers `(32, 16)`, ReLU activation, early stopping. |
| **Exp 8B** | **Deep Neural Network** | Deep Learning Classification | 4-layer Feedforward Network using TensorFlow/Keras: `Dense(32, relu) -> Dropout(0.2) -> Dense(16, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`. |
| **Exp 9** | **Ensemble Methods (RF & AdaBoost)** | Supervised Classification | Evaluated Bagging (**Random Forest**, 100 trees, max depth 6) vs Boosting (**AdaBoost**, 80 decision stumps, learning rate 0.5). |
| **Exp 10** | **Clustering (K-Means, GMM, Agglomerative)** | Unsupervised Patient Grouping | Evaluated K-Means, Agglomerative/Hierarchical Clustering, and Gaussian Mixture Models (GMM) using **Silhouette Score** and **Davies-Bouldin Index**. |

---

## 📊 3. Full Model Benchmark & Evaluation Results

All 7 classifiers were trained and tested on the exact same **stratified 80/20 train/test split** (239 training patients, 60 test patients):

| Rank | Model | Category | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **🏆 AdaBoost** | **Boosting Ensemble** | **81.67%** | **78.57%** | **57.89%** | **66.67%** | **0.8639** | **0.09s** |
| 2 | Random Forest | Bagging Ensemble | 78.33% | 68.75% | 57.89% | 62.86% | 0.8729 | 0.14s |
| 3 | SVM (RBF Kernel) | Kernel Method | 73.33% | 57.89% | 57.89% | 57.89% | 0.8177 | 0.01s |
| 4 | SVM (Linear Kernel) | Linear Classifier | 76.67% | 69.23% | 47.37% | 56.25% | 0.8614 | 0.02s |
| 5 | Deep Neural Network | Deep Learning | 71.67% | 55.56% | 52.63% | 54.05% | 0.7715 | 3.64s |
| 6 | KNN ($k=5$) | Instance-based | 71.67% | 58.33% | 36.84% | 45.16% | 0.6727 | 0.00s |
| 7 | MLP Classifier | Neural Network | 31.67% | 31.67% | 100.00% | 48.10% | 0.4082 | 0.02s |

---

## 🎯 4. Technical Justification: Why AdaBoost is the Best Model

When faculty ask: *"Why did you select AdaBoost?"*, explain these three technical reasons:

1. **Class Imbalance Handling:**
   - In the dataset, 203 patients (67.9%) survived while 96 patients (32.1%) passed away.
   - A naive baseline classifier guessing "Survived" for every patient achieves 67.9% accuracy but has zero utility.
   - AdaBoost achieved the highest **F1-Score (66.67%)** and **Precision (78.57%)**, ensuring low false alarm rates while catching true high-risk cases.
2. **Tabular vs Deep Learning Dynamics:**
   - Deep Neural Networks require tens of thousands of samples to learn weight representations without overfitting.
   - On a clean tabular dataset with 299 rows, decision-tree based boosting ensembles (AdaBoost) sequentially re-weight misclassified samples, making them mathematically superior for small-to-medium clinical data.
3. **Execution Efficiency:**
   - AdaBoost trains in under 0.1 seconds, making it lightweight, fast, and practical for deployment on local or edge clinical hardware.

---

## 🛠️ 5. Preprocessing & Feature Engineering

1. **Standardization:** All continuous numerical features were standardized using `StandardScaler`:
   $$z = \frac{x - \mu}{\sigma}$$
   This ensures features with large numerical magnitudes (e.g. CPK up to 7,861 mcg/L and platelets up to 850,000) do not dominate distance-based algorithms like KNN and gradient-based neural networks.
2. **Domain-Specific Feature Formulation:**
   - **Cardiorenal Ratio:** `serum_creatinine / ejection_fraction` — captures the combined failure of the cardiovascular and renal systems.
   - **Hyponatremia Flag:** `serum_sodium < 135 mEq/L` — clinical risk cutoff for electrolyte depletion.
   - **Severe Systolic Dysfunction Flag:** `ejection_fraction < 30%`.
   - **Geriatric Indicator:** `age >= 65`.
   - **Comorbidity Count:** `anaemia + diabetes + high_blood_pressure + smoking` (range 0 to 4).

---

## 👥 6. Unsupervised Learning: Clustering Analysis (Exp 10)

We applied clustering to discover natural patient groupings without using the `DEATH_EVENT` labels:

| Clustering Algorithm | Silhouette Score (Higher is better) | Davies-Bouldin Index (Lower is better) |
|---|:---:|:---:|
| **Agglomerative (Hierarchical)** | **0.1316** | **2.2658** |
| K-Means | 0.1056 | 2.4260 |
| Gaussian Mixture Model (GMM) | 0.0834 | 2.7036 |

### Clinical Interpretation of the 3 Clusters:
- **Group 0 (Compensated / Low Risk):** Normal ejection fraction (>45%), normal creatinine (<1.1 mg/dL). Actual mortality rate: **~12%**.
- **Group 1 (Cardiorenal Syndrome / High Risk):** Severely depressed ejection fraction (<30%), elevated serum creatinine (>1.8 mg/dL). Actual mortality rate: **~58%**.
- **Group 2 (Elderly Comorbid):** Older patients (mean age > 72) with multiple concurrent chronic diseases (hypertension + diabetes). Actual mortality rate: **~34%**.

---

## 💊 7. Collaborative Filtering: Recommendation System (Exp 6)

We implemented collaborative filtering using **Cosine Similarity**:
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

1. **Patient-Based (User-User CF):**
   - We construct an interaction matrix where rows are patients and columns are standard clinical interventions (ACEI/ARBs, Beta Blockers, SGLT2 Inhibitors, MRAs, Telemonitoring, Sodium Restriction).
   - Given a patient, the system computes cosine similarity against all historical records to find the top-$k$ most similar clinical cohort cases.
   - Recommended interventions are ranked by weighted average suitability scores of nearest neighbors.
2. **Intervention-Based (Item-Item CF):**
   - Transposes the matrix to find cosine similarity between medical interventions that frequently co-occur in successful management plans.

---

## 🖥️ 8. How the Streamlit Application Works

The user interface was built cleanly as an **end-user clinical decision support tool**:
- **Why are algorithm names hidden in the UI?** 
  Because in real hospital software, doctors do not interact with raw algorithm benchmarks or training scripts; they interact with patient assessment fields and clear clinical outputs. The machine learning pipeline operates robustly in the backend.
- **Three Core Workflow Pages:**
  1. **Patient Risk Assessment:** Input fields for age, EF, creatinine, sodium, CPK, platelets, comorbidities $\rightarrow$ Generates survival probability and mortality risk.
  2. **Patient Categories:** Clinical profiles describing the 3 patient cohorts discovered through unsupervised clustering.
  3. **Care Recommendations:** Patient cohort treatment suggestions powered by collaborative filtering.

---

## 🎤 9. Step-by-Step Live Demo Presentation Script

When demonstrating the project to faculty:

1. **Introduction (30 seconds):**
   > *"Good morning professors. For our Machine Learning Lab project, we chose the problem of Heart Failure Survival Prediction. Heart failure is characterized by acute decompensation where two biomarkers — Ejection Fraction and Serum Creatinine — are critical. We implemented and benchmarked all 7 classification algorithms from our lab experiments, evaluated clustering techniques, and built a collaborative filtering recommendation engine. Here is the working system."*

2. **Demonstrating Low-Risk Scenario (1 minute):**
   > *"First, let's test a stable patient: Age 52, Ejection Fraction 50% (healthy), Serum Creatinine 0.9 mg/dL (normal), Serum Sodium 140 mEq/L. When we run the assessment, the backend model outputs **Low Mortality Risk** with a high estimated survival probability (>85%)."*

3. **Demonstrating High-Risk Cardiorenal Scenario (1 minute):**
   > *"Now, let's observe a high-risk patient: Age 75, Ejection Fraction 20% (severe systolic heart failure), Serum Creatinine 2.4 mg/dL (renal impairment), Serum Sodium 130 mEq/L (hyponatremia), with hypertension. When we submit, the model instantly detects cardiorenal failure, outputting **High Mortality Risk (65-80%)** and highlighting the dangerously low EF and elevated creatinine."*

4. **Explaining Unsupervised Clustering & Recommendations (1 minute):**
   > *"Next, in our Patient Categories page, we applied unsupervised clustering (Agglomerative, K-Means, GMM) without using target labels, which naturally segregated patients into 3 clinical phenotypes. In our Care Recommendations page, we applied Collaborative Filtering with Cosine Similarity to suggest guideline-directed therapies based on similar patient cases."*

5. **Concluding with Algorithm Benchmark (1 minute):**
   > *"Across our experiments, AdaBoost achieved the best performance with an 81.67% accuracy and 66.67% F1-score, outperforming single models and deep neural nets on this tabular dataset."*

---

## ❓ 10. Top 15 Viva Questions & Model Answers

### Q1: What algorithm performed best and why?
> **Answer:** **AdaBoost** was our best model, achieving **81.67% accuracy, 66.67% F1-score, and 0.8639 ROC-AUC**. On structured tabular datasets with modest sample sizes (~300 records), boosting algorithms excel because they sequentially build weak learners (decision stumps) that focus on the hardest samples, preventing overfitting while handling non-linear interactions.

### Q2: Why did Deep Neural Networks and MLP score lower than AdaBoost?
> **Answer:** Deep Neural Networks and MLPs have a large number of parameters (weights and biases). Without tens of thousands of training instances, they are prone to overfitting or premature convergence on local minima, whereas decision tree ensembles have stronger inductive bias suited for small tabular data.

### Q3: Why is Accuracy an insufficient metric for this project?
> **Answer:** The dataset is imbalanced: 67.9% of patients survived while 32.1% died. A naive classifier predicting "Survived" for every patient gets 67.9% accuracy but fails to detect a single dying patient. In healthcare, False Negatives are fatal. Therefore, we used **F1-Score** (harmonic mean of precision and recall) and **ROC-AUC** as our primary selection criteria.

### Q4: What are the two most important features in predicting mortality?
> **Answer:** **Serum Creatinine** and **Ejection Fraction**. Ejection fraction measures the percentage of blood pumped by the left ventricle per contraction (contractility). Serum creatinine measures kidney filtration. When heart output drops, renal perfusion decreases, causing cardiorenal syndrome.

### Q5: How did you implement Collaborative Filtering (Exp 6)?
> **Answer:** We constructed a patient-treatment interaction matrix and calculated pairwise **Cosine Similarity**:
> $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
> For patient-based CF, unprescribed therapies were predicted using the weighted average indication scores of the top-5 nearest neighbor patients in the cohort.

### Q6: How does Random Forest differ from AdaBoost (Exp 9)?
> **Answer:** Both are ensemble methods, but **Random Forest uses Bagging** (Bootstrap Aggregation), where multiple deep decision trees are trained independently in parallel on random subsets of data and features, reducing variance. **AdaBoost uses Boosting**, where shallow decision stumps are trained sequentially, with each tree giving higher weight to samples misclassified by preceding trees, primarily reducing bias.

### Q7: Why did you use StandardScaler instead of MinMaxScaler?
> **Answer:** Features like CPK and Platelets contain natural clinical outliers. MinMaxScaler compresses all values strictly into $[0, 1]$, causing extreme outliers to squish the majority of normal data into a narrow range. StandardScaler centers features around mean 0 with variance 1 ($z = (x - \mu)/\sigma$), preserving relative distribution shapes better for algorithms like SVM and KNN.

### Q8: What was the difference between SVM Linear and SVM RBF (Exp 7)?
> **Answer:** SVM Linear finds an optimal linear separating hyperplane: $\vec{w} \cdot \vec{x} - b = 0$. SVM RBF uses the Radial Basis Function kernel:
> $$K(\vec{x}_i, \vec{x}_j) = \exp(-\gamma \|\vec{x}_i - \vec{x}_j\|^2)$$
> which implicitly maps clinical data into an infinite-dimensional feature space to construct non-linear decision boundaries. RBF achieved a better F1-score (57.89% vs 56.25%) because clinical biomarker interactions are non-linear.

### Q9: What is the Elbow method and Silhouette score in clustering (Exp 10)?
> **Answer:** The Elbow method plots within-cluster sum of squares (inertia) against $k$, looking for a bend where diminishing returns occur. The Silhouette score measures how similar an object is to its own cluster compared to other clusters (ranging from $-1$ to $+1$). Agglomerative clustering gave the best silhouette score (0.1316) for $k=3$.

### Q10: How does PCA differ from t-SNE (Exp 5)?
> **Answer:** PCA is a **linear** transformation that finds orthogonal axes (eigenvectors of the covariance matrix) that maximize explained variance. t-SNE is a **non-linear probabilistic** manifold technique that converts Euclidean distances between points into conditional probabilities and minimizes the Kullback-Leibler (KL) divergence in 2D space, better preserving local neighborhood clusters.

### Q11: Did you encounter any data leakage during preprocessing?
> **Answer:** No. We performed the train-test split *before* fitting the `StandardScaler`. The scaler was fitted exclusively on `X_train`, and then applied to transform both `X_train` and `X_test`. This ensures test set statistics never leaked into training.

### Q12: What does the ROC-AUC score represent?
> **Answer:** ROC-AUC measures the Area Under the Receiver Operating Characteristic Curve (plotting True Positive Rate vs False Positive Rate across all classification thresholds). A score of 0.5 represents random guessing; our best model scored **0.8639**, demonstrating strong ranking ability across decision thresholds.

### Q13: How did you handle hyperparameter tuning?
> **Answer:** In our training notebooks, we used cross-validation and hyperparameter searches (tuning $k$ and weights in KNN, $C$ and $\gamma$ in SVM, number of estimators and learning rate in AdaBoost, and tree depth in Random Forest).

### Q14: What is Cardiorenal Syndrome?
> **Answer:** It is a medical condition where dysfunction in either the heart or kidney induces dysfunction in the other. In our project, we engineered the feature `Renal_Cardiac_Ratio` ($serum\_creatinine / ejection\_fraction$) to represent this physiological interaction.

### Q15: Why is the Streamlit site designed for end-users instead of displaying benchmarks?
> **Answer:** In engineering practice, software deployment interfaces are designed for the end-user (clinicians), while technical benchmarks, loss curves, and algorithm comparisons belong in engineering reports and presentation decks. All models and benchmarks were saved in our `models/` directory and documented in this guide.

