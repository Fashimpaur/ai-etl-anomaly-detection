# 📘 Machine Learning Project – Notebook Reference & Technical Guide

This guide documents the full ML pipeline across the notebooks in this project.  
It explains **what each notebook does**, **why each step matters**, and **defines all machine-learning terms inline** the first time they appear.

The goal is to make the full workflow easy to understand, defend, and discuss with technical accuracy.

---

# Dataset Preparation & Preprocessing

The project starts by building a unified dataset from multiple raw files.  
A clean and consistent dataset is required before any machine learning model can operate.

### Key steps:

- Combining multiple raw exports into a single dataset  
- Standardizing column names  
- Parsing timestamps into usable date/time formats  
- Converting categorical variables (text) into numerical encodings  
- Scaling numeric variables to bring them onto similar ranges  
- Handling missing values in a principled, statistical way  

These steps form the *foundation* of every later model.

---

# Feature Engineering

Feature engineering transforms raw columns into structured numerical features suitable for machine learning algorithms.

Below are the transformations used in the notebooks, including inline definitions and formulas.

---

## Scaling Numerical Features

Many ML models assume or benefit from numeric features that are scaled in consistent ways.  
Scaling prevents large-valued columns from dominating training.

Two primary scalers are used:

---

### 🔹 StandardScaler

**Purpose:**  
Transforms a numeric feature so that its mean becomes **0** and its standard deviation becomes **1**.  
This is appropriate when the data distribution is roughly bell-shaped (normal).

**Formula:**

\[
z = \frac{x - \mu}{\sigma}
\]

Where:  
- **x** = original value  
- **μ (mu)** = feature mean  
- **σ (sigma)** = feature standard deviation  
- **z** = standardized value  

**Effect:**  
Values are centered around zero and expressed in units of standard deviations.

---

### 🔹 MinMaxScaler

**Purpose:**  
Transforms a numeric feature into a fixed range, usually **0 to 1**.  
This preserves the shape of the distribution while normalizing scale.

**Formula:**

\[
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]

Where:  
- **x** = original value  
- **xmin** = minimum observed value  
- **xmax** = maximum observed value  
- **xscaled** = normalized value  

**Effect:**  
Useful when a bounded range is required (e.g., distance-based models like k-NN, or neural networks).

---

## Encoding Categorical Features

ML models cannot operate directly on text values such as `"status" = 'Approved'`.  
These categorical features must be converted into numbers.

Two approaches are used depending on the semantics of the feature:

---

### 🔹 One-Hot Encoding

**Purpose:**  
For categories that have *no natural order* (e.g., Red, Blue, Green), one-hot encoding creates a separate binary column for each category.

Example:

“Color” →  
- `Color_Red`  
- `Color_Green`  
- `Color_Blue`

Each row gets **1** where its category is present, otherwise **0**.

**Why it's used:**  
Prevents algorithms from incorrectly treating categories as numeric quantities.

---

### 🔹 Ordinal Encoding

**Purpose:**  
Used when categories *do have an inherent order* (e.g., Small < Medium < Large).

Example:

Small → 0  
Medium → 1  
Large → 2

**Why it's used:**  
Preserves the natural ranking, which is useful for tree-based models and compact numeric representations.

---

## Handling Missing Data

Real-world datasets often contain missing values.  
This project uses practical statistical imputation strategies:

- **Mean imputation** for normally-distributed numeric features  
- **Median imputation** for skewed numeric features  
- **Most-frequent category imputation** for categorical columns  
- **Row dropping only when absolutely necessary** to avoid biasing the model  

---

# Anomaly Detection

This portion of the notebooks focuses on identifying values that are unusual compared to the general population of data.

---

## What Is Anomaly Detection?

**Anomaly detection** is the task of finding data points that differ significantly from typical patterns.  
Anomalies can indicate:

- fraud  
- system misuse  
- outliers or corrupted data  
- rare edge cases  
- unexpected shifts in behavior  

Anomalies are usually rare and often unlabeled, making this problem suitable for **unsupervised learning** methods.

_Unsupervised learning_ means the algorithm finds structure in the data **without access to labels** indicating what is “normal” or “anomalous.”

---

## Isolation Forest

Isolation Forest is the primary anomaly detection model in the project.  
It is efficient, interpretable, and works well in high-dimensional settings.

### Core Concept  
Anomalies are easier to isolate because they appear far from dense clusters of normal points.

### How the algorithm works:

1. Randomly choose a feature  
2. Randomly choose a split point  
3. Split the dataset  
4. Repeat recursively until a specific data point is isolated  

This generates a **path length**—the number of splits needed to isolate the point.

- **Short path length → likely anomaly**  
- **Long path length → likely normal**

### Scoring Formula:

**$$
s(x) = 2^{-\frac{E(h(x))}{c(n)}}
$$**

Where:

- **s(x)** = anomaly score (closer to 1 means more anomalous)  
- **E(h(x))** = expected path length of sample x  
- **c(n)** = normalization factor based on dataset size  
- **n** = number of samples  

Isolation Forest is especially good at:

- detecting rare, irregular patterns  
- handling non-linear data  
- scaling to large datasets quickly  

---

# 4. Model Training & Evaluation

This section describes the supervised and unsupervised modeling processes implemented in the notebooks.

---

## Splitting the Data

All modeling begins with dividing the dataset into two parts:

- **Training set** → used to fit the model  
- **Testing set** → used to evaluate it on unseen data  

Typical split: **80% train / 20% test**

This helps measure how well the model generalizes beyond the training data.

---

## Models Used

### Supervised Models (when labels are available):

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

Supervised learning means the model is trained using known “correct answers,” such as a column named `label` with values 0 (normal) and 1 (anomaly).

---

### Unsupervised Models (when labels are missing):

- Isolation Forest  
- Local Outlier Factor  
- Autoencoders  

Unsupervised models discover structure without guidance.

---

## Evaluation Metrics

All classification models rely on the following foundational metrics:

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}
$$

### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Score

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Where:

- **TP** = true positives  
- **TN** = true negatives  
- **FP** = false positives  
- **FN** = false negatives  

These allow comparison between different models and ensure performance is interpreted correctly.

---

# Data Visualization

The notebooks also provide visual analysis to make results interpretable.

### KDE Plots (Kernel Density Estimates)  
These show smooth approximations of data distributions.  
For example, comparing the distribution of transaction amounts for "normal" vs. "anomaly" cases.

A KDE plot gives more insight than a simple histogram because it provides a continuous curve representing likelihood density.

### Scatterplots & Correlation Heatmaps  
Used to identify:

- linear relationships  
- clusters  
- outliers  
- correlations between features  

---

# Summary

This notebook README provides a complete, technically rigorous description of the ML pipeline used in this project:

- How data is cleaned and prepared  
- How scaling and encoding are applied  
- Why particular algorithms were selected  
- How anomaly detection works  
- How Isolation Forest isolates rare patterns  
- How models are trained, validated, and scored  
- How visualizations support interpretation
