# 🏆 Fault Detection Classification - Competition Solution

## 📌 Overview

This repository presents a complete end-to-end Machine Learning solution for a binary classification competition focused on detecting faulty system conditions from anonymized numerical sensor features.

The objective is to classify each sample as:

- **0 → Normal**
- **1 → Faulty**

The pipeline covers:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model benchmarking  
- Hyperparameter tuning  
- Model interpretability  
- Test prediction generation  
- Competition-ready submission file creation  

The final output is a `submission.csv` file formatted as required for evaluation.

---

## 🎯 Competition Objective

Given:

- `TRAIN.csv` → Feature set + target column `Class`
- `TEST.csv` → Feature set + `ID` column

The task is to:

1. Train a classification model using the training data.
2. Predict the target class for the test dataset.
3. Generate a submission file in the format:

```
ID, CLASS
```

### Optimization Priority

- **Primary Metric:** F1-Score  
- **Secondary Metric:** ROC-AUC  

Since this is a fault detection problem, minimizing **false negatives** is critical.

---

## 📂 Dataset Description

### 🔹 Training Data (`TRAIN.csv`)
- Features: `F01` – `F47`
- Target column: `Class`

### 🔹 Test Data (`TEST.csv`)
- Features: `F01` – `F47`
- Identifier column: `ID`

### 🔹 Target Classes

| Class | Meaning  |
|--------|----------|
| 0      | Normal   |
| 1      | Faulty   |

Stratified sampling is used to preserve class distribution during validation.

---

## 🧠 Solution Methodology

### 1️⃣ Data Preprocessing

- Checked for missing values
- Applied median imputation (if required)
- Used **RobustScaler** to handle outliers effectively
- Handled class imbalance using `class_weight='balanced'`

---

### 2️⃣ Exploratory Data Analysis (EDA)

Performed:

- Target distribution visualization
- Feature distribution comparison (Normal vs Faulty)
- Correlation heatmap analysis
- High-variance feature inspection
- Boxplots and histogram comparisons

Purpose:
- Identify separability patterns
- Detect correlations
- Understand data imbalance

---

### 3️⃣ Feature Engineering

To enhance predictive power, additional statistical features were created.

#### 🔹 Group-Based Statistical Features

For logical feature groups:

- Mean  
- Standard Deviation  
- Maximum  
- Minimum  
- Range  

This captures aggregated behavior across related feature blocks.

#### 🔹 Ratio Features

Created feature ratios relative to `F10` to capture relative scale differences and interaction effects.

This significantly improved model performance and separability.

---

### 4️⃣ Model Benchmarking

The following models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (if available)

Each model was implemented inside a **Scikit-learn Pipeline** to prevent data leakage.

#### Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

ROC curves and comparative bar charts were generated for analysis.

---

### 5️⃣ Hyperparameter Tuning

The best baseline model was:

> **Random Forest Classifier**

Tuning strategy:

- `RandomizedSearchCV`
- Stratified 3-Fold Cross Validation
- Optimized for F1-score
- 20 randomized parameter combinations

#### Tuned Parameters:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

This improved generalization and validation performance.

---

### 6️⃣ Model Interpretability

Feature importance was extracted from the tuned Random Forest model.

Analysis included:

- Top 20 most important features
- Cumulative importance curve
- Identification of number of features required to reach 95% importance

This ensures model transparency and explainability.

---

## 📊 Evaluation Strategy

### Train-Validation Split

- 80% Training
- 20% Validation
- Stratified sampling to preserve class balance

### Validation Analysis Included:

- Confusion Matrix
- ROC Curve
- Precision-Recall balance
- Prediction probability distribution

Special focus was placed on:

- Minimizing False Negatives
- Maintaining strong F1-score
- Preserving ROC-AUC robustness

---

## 🚀 End-to-End Pipeline Architecture

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train-Validation Split (Stratified)
   ↓
Model Benchmarking
   ↓
Hyperparameter Optimization
   ↓
Feature Importance Analysis
   ↓
Test Prediction
   ↓
submission.csv
```

---

## 📦 Installation

Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## ▶️ How to Run

1. Place `TRAIN.csv` and `TEST.csv` in the root directory.
2. Execute:

```bash
python ML Challenge.ipynb
```

3. After successful execution, the following file will be generated:

```
submission.csv
```

---

## 📁 Project Structure

```
├── TRAIN.csv
├── TEST.csv
├── ML Challenge.ipynb
├── FNAL.csv
└── README.md
```

---

## 🏁 Submission Format

The generated submission file follows:

```
ID, CLASS
10001, 0
10002, 1
10003, 0
...
```

Where:

- `0 → Normal`
- `1 → Faulty`

---

## 🔥 Key Strengths of the Solution

- Fully automated end-to-end ML pipeline
- Robust scaling for outlier handling
- Feature-engineered statistical enhancements
- Class imbalance handling
- Cross-validated hyperparameter tuning
- Model interpretability included
- Competition-ready output format

---

## 📌 Conclusion

This solution emphasizes:

- Strong generalization
- Balanced precision-recall tradeoff
- High F1-score under class imbalance
- Transparent and explainable modeling

The modular pipeline can be extended with:

- Ensemble stacking
- Threshold optimization
- Advanced feature selection
- Deployment integration (Flask / API)

