# AI-Based Diabetes Risk Prediction Dashboard

**IBM SkillsBuild AI/ML Internship — Project 1**  
*B.Pharm Student Educational Machine Learning Project*

---

## ⚠️ Educational Disclaimer

This project was built as part of an academic internship. All predictions are **model estimates based on statistical patterns** in training data. This tool is **not** a medical diagnostic instrument and **must not** be used to make any health-related decisions. Always consult a licensed healthcare professional.

---

## 📌 Project Overview

This project demonstrates how machine learning can be used to classify health records into two categories — diabetic (positive) and non-diabetic (negative) — using supervised learning algorithms. The project includes a complete training pipeline and an interactive web dashboard built with Streamlit.

---

## 🎯 Problem Statement

Diabetes is a chronic metabolic condition affecting hundreds of millions of people worldwide. Early identification of high-risk individuals using routine health data can support preventive care. This project explores whether machine learning models can classify records based on common health indicators.

---

## 🏆 Project Objectives

1. Explore and understand the diabetes prediction dataset.
2. Train and evaluate two machine learning models.
3. Handle class imbalance appropriately.
4. Select the best model based on meaningful evaluation metrics.
5. Build an interactive dashboard for demonstration purposes.
6. Document the entire process clearly for a beginner audience.

---

## 📂 Dataset Information

**Source:** [Kaggle — Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

**File:** `data/diabetes_prediction_dataset.csv`

| Column | Type | Description |
|---|---|---|
| `gender` | Categorical | Biological sex: Female, Male, Other |
| `age` | Numerical | Age in years (0.08 – 80) |
| `hypertension` | Binary (0/1) | High blood pressure: 0 = No, 1 = Yes |
| `heart_disease` | Binary (0/1) | Heart disease history: 0 = No, 1 = Yes |
| `smoking_history` | Categorical | never, No Info, current, former, ever, not current |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | Glycated haemoglobin (%) — 3-month avg blood sugar |
| `blood_glucose_level` | Numerical | Current blood glucose in mg/dL |
| `diabetes` | Target (0/1) | **0 = Negative**, **1 = Positive** |

**Dataset statistics:**
- Total records: 100,000
- Duplicate rows: 3,854 (removed before training)
- Missing values: None
- Class distribution: 91,500 negative (0) vs 8,500 positive (1)

---

## 🗂️ Folder Structure

```
PROJECT1/
│
├── data/
│   └── diabetes_prediction_dataset.csv   ← Original dataset (do not modify)
│
├── models/
│   ├── best_pipeline.joblib              ← Saved trained pipeline (generated after training)
│   └── model_metrics.json               ← Evaluation results (generated after training)
│
├── .env.example                          ← Template for environment variables (no secrets needed)
├── agent_instructions.md                 ← Instructions for future AI coding assistance
├── app.py                                ← Streamlit dashboard
├── README.md                             ← This file
├── requirements.txt                      ← Python package dependencies
└── train_model.py                        ← Model training and evaluation script
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, inspection, and manipulation |
| **NumPy** | Numerical operations |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **Streamlit** | Interactive web dashboard |
| **Matplotlib** | Data visualizations and charts |
| **Joblib** | Saving and loading the trained pipeline |

No cloud services, APIs, or paid tools are required. The project runs entirely on your local computer.

---

## 🔧 Data Preprocessing

1. **Duplicate removal** — 3,854 duplicate rows are removed before training.
2. **Train/test split** — 80% training, 20% test, stratified by class label, fixed random seed (42).
3. **Numerical features** (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`) — scaled with `StandardScaler` to mean=0, std=1.
4. **Binary features** (`hypertension`, `heart_disease`) — passed through unchanged (already 0/1).
5. **Categorical features** (`gender`, `smoking_history`) — encoded with `OneHotEncoder`.
6. All preprocessing steps are fitted **only on training data** using a `Pipeline` + `ColumnTransformer` to prevent data leakage.

---

## 🤖 Algorithms Used

### Logistic Regression
- A simple, interpretable linear classification algorithm.
- Estimates the probability of belonging to the positive class.
- Good baseline for binary classification.

### Random Forest Classifier
- An ensemble of decision trees — each tree votes on the class.
- Handles non-linear relationships better than logistic regression.
- Generally more powerful for tabular health data.

**Why `class_weight='balanced'`?**  
The dataset is imbalanced: 91.5% negative, 8.5% positive. Without correction, a model could achieve 91.5% accuracy by predicting "negative" for every record. Using `class_weight='balanced'` tells the algorithm to give more importance to the minority class (positive), improving recall on diabetic records.

---

## 📊 Evaluation Metrics

| Metric | Why it matters |
|---|---|
| **Accuracy** | % of all records classified correctly — can be misleading with imbalanced data |
| **Precision** | Of all predicted positives, how many are truly positive |
| **Recall** | Of all actual positives, how many did the model catch — critical for health screening |
| **F1-Score** | Harmonic mean of Precision and Recall — the primary selection criterion |
| **ROC-AUC** | Measures overall discriminative ability (1.0 = perfect) |
| **Confusion Matrix** | Shows TP, TN, FP, FN counts clearly |

**Model selection criterion:** F1-score for the positive class (diabetes = 1). Accuracy alone is not used because the dataset is imbalanced.

---

## 💻 Installation Instructions

### Step 1 — Make sure Python is installed
Check your Python version (3.9 or later is recommended):
```
python --version
```

### Step 2 — Navigate to the PROJECT1 folder
```
cd PROJECT1
```

### Step 3 — Install all required packages
```
python -m pip install -r requirements.txt
```

---

## 🚀 How to Run

### Train the models
Run this **once** before starting the dashboard:
```
python train_model.py
```

This script will:
- Load the dataset and print an inspection report
- Remove duplicates
- Split data into training and test sets
- Train Logistic Regression and Random Forest models
- Print evaluation metrics for both models
- Save the best pipeline to `models/best_pipeline.joblib`
- Save the metrics to `models/model_metrics.json`

### Start the dashboard
```
python -m streamlit run app.py
```

The dashboard will open automatically in your web browser at:  
**http://localhost:8501**

If it does not open automatically, copy and paste the URL into your browser.

---

## 📋 Expected Outputs

After running `train_model.py`, you will see:
- Dataset shape, dtypes, missing values, duplicates, and class distribution
- Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix for both models
- A statement of which model was selected and why

After running `app.py`, you will see a dashboard with:
- Project introduction and disclaimer
- Dataset charts (class distribution, age/BMI/HbA1c distributions, smoking history)
- Model comparison table and confusion matrices
- A prediction form where you can enter values and get a class prediction

*(Add screenshots here after running the project)*

---

## ⚠️ Limitations and Ethical Considerations

1. **Not a medical tool.** The model is trained on patterns in a publicly available dataset and cannot replace clinical examination.
2. **Dataset bias.** The dataset is heavily imbalanced (91.5% negative). Class weighting partially addresses this, but the model may still have higher false-negative rates.
3. **Dataset source unknown.** Kaggle datasets may not reflect all real-world populations.
4. **No feature importance explanation.** The dashboard does not currently provide explanations for individual predictions.
5. **No deployment.** This project is designed to run locally only.
6. **Do not use on real patients.** This tool must not be used in any clinical or diagnostic setting.

---

## 📤 GitHub Upload Instructions

### Step 1 — Create a .gitignore file
Before uploading, create a file named `.gitignore` in the PROJECT1 folder with this content:
```
models/
__pycache__/
*.pyc
.env
*.joblib
```
This prevents the trained model file and any sensitive files from being uploaded.

### Step 2 — Initialize a local Git repository
```
git init
git add .
git commit -m "Initial commit: AI Diabetes Risk Prediction Project"
```

### Step 3 — Create a new repository on GitHub
Go to https://github.com/new, create a new public repository.

### Step 4 — Push your code
```
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

**Important:** Do not upload the `models/` folder or any `.env` file containing real credentials.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `python -m pip install -r requirements.txt` |
| `FileNotFoundError: diabetes_prediction_dataset.csv` | Ensure the CSV is inside the `data/` folder |
| Dashboard shows "Model files not found" | Run `python train_model.py` first |
| `streamlit` command not found | Use `python -m streamlit run app.py` instead |
| Charts not displaying | Ensure `matplotlib` is installed |
| Slow training | Random Forest with 100 trees on 100k rows may take 1–3 minutes on older hardware — this is normal |

---

## 🌐 Optional Deployment

To share your dashboard publicly, you can deploy it for free on **Streamlit Community Cloud**:

1. Push your project to GitHub (see above).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app**, select your repository and `app.py` as the entry file.
4. Click **Deploy**.

Note: You will need to either commit the trained model files or add a startup script that runs `train_model.py` before the app starts.
