# 🧠 Mental Health & Depression Risk Classifier

This repository contains a machine learning-based classification system and interactive dashboard for predicting depression risk. The project analyzes demographic details, lifestyle regulators, clinical history, and psychological stressors to perform risk assessments.

---

## 🚀 Project Overview

The project is structured into two main components:
1. **Interactive Dashboard (`app.py`)**: A premium, dark-themed Streamlit application that allows users to input demographic, lifestyle, and stress indicators to receive a real-time depression risk assessment with predicted probabilities.
2. **Model Development (`Random_Forest_MH.ipynb`)**: A Jupyter notebook detailing the Exploratory Data Analysis (EDA), data cleaning, handling of missing values, feature engineering (one-hot encoding), and model training/evaluation.

---

## 🛠️ Algorithm & Model Performance

The depression risk classification model is trained using an **Ensemble Random Forest Classifier** (`scikit-learn`).

### Model Configuration
- **Algorithm**: `RandomForestClassifier`
- **Number of Estimators (`n_estimators`)**: `100`
- **Max Depth (`max_depth`)**: `15`
- **Random State (`random_state`)**: `42`
- **Parallel Jobs (`n_jobs`)**: `-1` (utilizes all available CPU cores)

### Performance Evaluation
The model was trained on a split of **112,560 training samples** and evaluated on **28,140 validation samples** (20% split):
- **Accuracy Score**: `92.40%`
- **ROC AUC Score**: `96.45%`

#### Classification Report:
| Class | Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | Not Depressed | `0.94` | `0.97` | `0.95` | 23,027 |
| **1** | Depressed | `0.84` | `0.71` | `0.77` | 5,113 |
| **Accuracy** | | | | `0.92` | 28,140 |

---

## 📦 Dependencies

The project is configured to use Python `3.13` and is managed using the [uv package manager](https://github.com/astral-sh/uv). The core dependencies specified in `pyproject.toml` are:

- **Python**: `>=3.13`
- **scikit-learn**: `>=1.9.0` (for machine learning pipelines and the Random Forest model)
- **streamlit**: `>=1.59.2` (for the web application dashboard)
- **pandas**: `>=3.0.3` (for data cleaning and manipulation)
- **numpy**: `>=2.5.1` (for numerical computations)
- **seaborn**: `>=0.13.2` (for visualization in the EDA notebook)

*Note: The pre-trained model is serialized using `joblib`.*

---

## 📂 File Structure

```
Mental_Health/
├── .gitignore                # Specifies untracked files to ignore (e.g., venv, local cache)
├── .python-version           # Recommended Python environment version (3.13)
├── pyproject.toml            # Project configuration and dependency declarations
├── uv.lock                   # Lockfile containing exact dependency versions for reproducible builds
├── train.csv                 # Raw dataset containing 140,700 samples and 20 features
├── Random_Forest_MH.ipynb    # Jupyter Notebook for EDA, data preprocessing, and model training
├── random_forest_model.pkl   # Serialized pre-trained Random Forest model
├── model_features.json       # JSON file containing the exact feature list for model alignment
├── app.py                    # Streamlit web app providing the interactive dashboard
├── main.py                   # Boilerplate project entrypoint
└── README.md                 # Project documentation (this file)
```

---

## 💻 How to Run the App

1. Ensure you have Python `3.13` and your dependencies installed. If using `uv`, you can install dependencies and run:
   ```bash
   uv pip install -r pyproject.toml
   ```
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the local address (typically `http://localhost:8501`) in your browser to interact with the dashboard.
