# 🧠 Parkinson's Disease Detection — ML Desktop App

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-97--99%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A machine learning desktop application that detects Parkinson's Disease from **22 vocal biomarker measurements** using a **Random Forest Classifier**. Built with Python and Tkinter, the app provides an interactive GUI for manual input, quick sample loading, and random value generation with instant AI-powered predictions.

---

## 📸 Features

- 🖥️ **Full Desktop GUI** — Dark-themed professional interface built with Tkinter
- 🎯 **Random Forest ML Model** — Trained on vocal biomarker data with 97–99% accuracy
- 🎲 **Random Test Mode** — Auto-generates random values and instantly predicts the result
- 📋 **Sample Data Loaders** — One-click fill with healthy or Parkinson's patient samples
- 📊 **Confidence Probability Bar** — Shows prediction certainty split between Healthy vs Parkinson's
- 🔬 **22 Vocal Biomarkers** — Organized into Frequency, Jitter, Shimmer, Noise, and Nonlinear sections
- 🗃️ **100,000 Row Synthetic Dataset** — Statistically generated from real patient data

---

## 📁 Project Structure

```
parkinsons-detector/
│
├── parkinsons_app.py          # Main desktop application (GUI + ML model)
├── parkinsons.csv             # Dataset (original 195 rows or synthetic 100k)
├── parkinsons_100k.csv        # Synthetic 100,000 patient dataset
├── parkinsons_sample.xlsx     # Sample test data with instructions
├── Random Forest 5 Trees.png  # Visualization of first 5 decision trees (auto-generated)
└── README.md                  # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/parkinsons-detector.git
cd parkinsons-detector
```

### Step 2 — Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Step 3 — Run the App

```bash
python parkinsons_app.py
```

> ⚠️ Make sure `parkinsons.csv` is in the **same folder** as `parkinsons_app.py` before running.

---

## 🧪 How to Test

### Option A — Quick Fill Buttons (Easiest)
Click **"Load Healthy Sample"** or **"Load Parkinson Sample"** in the sidebar. All 22 fields will auto-fill. Then click **RUN ANALYSIS**.

### Option B — Random Auto-Predict
Click **"🎲 Random & Auto-Predict"**. The app generates statistically valid random values and immediately shows the result — no manual input needed.

### Option C — Manual Entry
Enter all 22 vocal biomarker values manually into the fields (valid ranges shown under each field) and click **RUN ANALYSIS**.

### Option D — Use the Sample Excel File
Open `parkinsons_sample.xlsx`, copy any row's 22 feature values (columns D to X), and paste them into the app fields.

---

## 🌲 How the Random Forest Classifier Works

### What is a Random Forest?

A **Random Forest** is an ensemble learning algorithm — it builds many individual **Decision Trees** and combines their predictions to produce a more accurate and stable result. The name comes from the idea of growing a "forest" of trees where each tree is trained on a slightly different version of the data.

```
Input Data (22 features)
        │
        ▼
┌───────────────────────────────────────┐
│           RANDOM FOREST               │
│                                       │
│  Tree 1   Tree 2   Tree 3  ...  Tree N│
│    │         │        │           │   │
│  "Yes"     "No"    "Yes"  ...  "Yes"  │
└───────────────────────────────────────┘
        │
        ▼
  MAJORITY VOTE → Final Prediction
```

### Step-by-Step: How This Project Uses It

#### 1. Data Loading & Feature Extraction
```python
df = pd.read_csv("parkinsons.csv")
features = df.drop(['name', 'status'], axis=1)  # 22 vocal features
target   = df['status']                          # 0 = Healthy, 1 = Parkinson's
```
The `name` column is an identifier (not useful for prediction) and `status` is what we're trying to predict, so both are removed from the input features.

#### 2. Feature Scaling with MinMaxScaler
```python
scaler = MinMaxScaler((-1, 1))
features_scaled = scaler.fit_transform(features)
```
All 22 features are scaled to the range **[-1, 1]**. This is important because the features have vastly different scales — for example `MDVP:Fo(Hz)` ranges from 88 to 260, while `MDVP:Jitter(Abs)` ranges from 0.000007 to 0.00026. Without scaling, larger-valued features would unfairly dominate the model.

#### 3. Train/Test Split
```python
x_train, x_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=10
)
```
80% of the data is used to **train** the model. 20% is held back as a **test set** that the model has never seen, used to evaluate real-world accuracy.

#### 4. Building Each Decision Tree

A single Decision Tree works by asking a series of yes/no questions about the features:

```
Is MDVP:Fo(Hz) < 0.3 ?
├── YES → Is NHR > -0.5 ?
│         ├── YES → Parkinson's ✓
│         └── NO  → Healthy ✓
└── NO  → Is PPE > 0.2 ?
          ├── YES → Parkinson's ✓
          └── NO  → Healthy ✓
```

At each node, the tree finds the **best feature and threshold** that splits the data to separate Parkinson's patients from healthy ones as cleanly as possible. This is measured using **Gini Impurity** — a score of 0 means perfect separation (all one class), a score of 0.5 means maximum mixing.

#### 5. The "Random" Part — Bootstrap + Feature Randomness

What makes a Random Forest different from just one big decision tree is two sources of randomness:

**Bootstrap Sampling (Bagging):**
Each tree is trained on a random sample of the training data (with replacement). Roughly 63% of rows are included in each tree's training set; the rest are left out. This means each tree sees a slightly different version of the dataset, so all trees are different from each other.

**Random Feature Selection:**
At each split point, instead of considering all 22 features, each tree only considers a random subset (typically √22 ≈ 5 features). This forces the trees to be diverse and prevents them all from making the same splits.

```
Original Dataset (195 rows × 22 features)
│
├── Tree 1 trains on → rows [3,7,1,1,12,8,...] features [Fo, Jitter, HNR, PPE, ...]
├── Tree 2 trains on → rows [2,9,9,4,15,3,...] features [Shimmer, NHR, RPDE, DFA, ...]
├── Tree 3 trains on → rows [6,1,14,2,10,5,...] features [Fhi, APQ, spread1, D2, ...]
└── ...  (100 trees by default)
```

#### 6. Prediction by Majority Vote

When predicting a new patient, the input is passed through all 100 trees. Each tree independently outputs a vote: **0 (Healthy)** or **1 (Parkinson's)**. The final prediction is whichever class gets **more than 50% of the votes**.

```python
y_pred = model.predict(x_test)
```

The `predict_proba()` method returns the actual vote percentages:
```python
proba = model.predict_proba(input)[0]
# e.g., [0.08, 0.92] → 8% Healthy, 92% Parkinson's
```
This is shown as the **confidence probability bar** in the app's result popup.

#### 7. Tree Visualization
The app auto-generates a visualization of the first 5 trees in the forest:
```python
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(model.estimators_[index],
                   feature_names=features.columns,
                   class_names=['Healthy', 'Parkinson'],
                   filled=True,
                   ax=axes[index])
fig.savefig('Random Forest 5 Trees.png')
```
Each colored box in the visualization represents a decision node — blue boxes lean toward Healthy, orange boxes lean toward Parkinson's. The shade intensity represents how pure (confident) that node is.

---

## 📊 The Dataset

### Original Dataset
- **Source:** UCI Machine Learning Repository — Parkinson's Disease Dataset
- **Patients:** 195 voice recordings from 31 people (23 with Parkinson's, 8 healthy)
- **Features:** 22 vocal biomarker measurements
- **Class balance:** 147 Parkinson's (75.4%) | 48 Healthy (24.6%)

### Synthetic 100,000 Row Dataset

The `parkinsons_100k.csv` file was generated using **Multivariate Normal Sampling** — a statistically rigorous method that preserves the original data's structure:

```python
from scipy.stats import multivariate_normal

# Learn the statistical structure of each class separately
mean = data.mean(axis=0)   # Mean of each feature
cov  = np.cov(data.T)      # Full 22×22 covariance matrix

# Generate new samples following the same distribution
samples = multivariate_normal.rvs(mean=mean, cov=cov, size=n)

# Clip to observed real-world bounds
samples = np.clip(samples, data.min(axis=0), data.max(axis=0))
```

**Why this approach is better than pure random:**
- A covariance matrix captures how features **relate to each other** — e.g., in Parkinson's patients, Jitter and Shimmer tend to be elevated together. Simple random generation would break these relationships.
- The class ratio (75.4% / 24.6%) is preserved exactly.
- All values are clipped to stay within the medically observed min/max bounds.

| Property | Original | Synthetic 100k |
|---|---|---|
| Total rows | 195 | 100,000 |
| Parkinson's | 147 (75.4%) | 75,400 (75.4%) |
| Healthy | 48 (24.6%) | 24,600 (24.6%) |
| Feature correlations | ✅ Real | ✅ Preserved |
| Value bounds | ✅ Real | ✅ Clipped to real |

---

## 🎯 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 97–99% |
| Algorithm | Random Forest Classifier |
| Trees | 100 (default) |
| Train/Test Split | 80% / 20% |
| Feature Scaling | MinMaxScaler [-1, 1] |

---

## 🔬 The 22 Vocal Biomarkers

Parkinson's Disease affects the muscles that control speech, producing measurable changes in voice recordings. The 22 features are grouped into 5 categories:

| Category | Features | What It Measures |
|---|---|---|
| **Frequency** | MDVP:Fo, Fhi, Flo | Average, max, and min fundamental vocal frequency in Hz |
| **Jitter** | Jitter(%), Jitter(Abs), RAP, PPQ, DDP | Cycle-to-cycle variation in vocal frequency — higher in Parkinson's |
| **Shimmer** | Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA | Cycle-to-cycle variation in vocal amplitude — higher in Parkinson's |
| **Noise** | NHR, HNR | Ratio of noise to tonal components in the voice |
| **Nonlinear** | RPDE, DFA, spread1, spread2, D2, PPE | Nonlinear dynamical complexity measures of vocal signal |

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.21 | Numerical computation |
| `pandas` | ≥1.3 | Data loading and manipulation |
| `scikit-learn` | ≥0.24 | Random Forest model, scaler, metrics |
| `matplotlib` | ≥3.4 | Decision tree visualization |
| `tkinter` | Built-in | Desktop GUI framework |

Install all at once:
```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 💡 Tips for Higher Accuracy

If you want to push accuracy above 99%, try these modifications in `parkinsons_app.py`:

```python
# More trees + tuned parameters
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=2,
    n_jobs=-1   # Uses all CPU cores for faster training
)
```

Or switch to XGBoost which often performs slightly better on medical datasets:
```bash
pip install xgboost
```
```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=2)
```

---

## ⚕️ Disclaimer

This application is built for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. If you or someone you know is experiencing symptoms of Parkinson's Disease, please consult a qualified medical professional.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository — Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) by Max Little, University of Oxford
- Original paper: *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection* — Little MA et al., 2007

---

*Built with ❤️ as part of ProjectGurukul*
