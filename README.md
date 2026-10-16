# Customer Churn Prediction - End-to-End Machine Learning System

A complete end-to-end Machine Learning system and Flask web application for **Telco Customer Churn Prediction** based on the dataset analysis in `notebooks/Customer_Churn.ipynb` and `customer_churn.py`.

The system includes automated data preprocessing, Yeo-Johnson power transformation, StandardScaler normalization, One-Hot Encoding, SMOTE class balancing, hyperparameter tuning via GridSearchCV, a Flask backend REST API, and a clean, responsive enterprise web dashboard.

---

## 📁 Repository Structure

```
Customer-Churn-Prediction/
├── artifacts/                          # Machine Learning Model & Preprocessor Artifacts
│   ├── model.pkl                       # Trained SMOTE-balanced Logistic Regression pipeline
│   ├── scaler.pkl                      # Fitted StandardScaler (tenure, MonthlyCharges)
│   ├── power_transformer.pkl           # Fitted Yeo-Johnson PowerTransformer (TotalCharges)
│   ├── feature_names.json              # List of all 22 final input features
│   └── batch_predictions_output.csv    # Generated batch prediction output CSV
├── datasets/                           # Project Datasets Folder
│   ├── Telco-Customer-Churn.csv        # Raw Telco dataset
│   └── cleaned_dataset.csv             # Cleaned & processed dataset
├── notebooks/                          # Research & Exploration Notebooks
│   └── Customer_Churn.ipynb            # Jupyter Notebook with EDA & model trials
├── src/                                # Modular Machine Learning Engine
│   ├── __init__.py
│   ├── data_preprocessing.py           # Preprocessing & transformation module
│   ├── model_training.py               # Model training script with SMOTE & tuning
│   └── predict_pipeline.py             # Inference pipeline for single & batch inputs
├── app.py                              # Flask Web Application & REST API Server
├── templates/
│   └── index.html                      # Enterprise Web Dashboard (Toggle Switches & 3 Tabs)
├── static/
│   ├── css/
│   │   └── style.css                   # Enterprise Light Dashboard Styling
│   └── js/
│       └── main.js                     # Tab switching, AJAX prediction & CSV download logic
├── customer_churn.py                   # Reference Python script
├── requirements.txt                    # Project Python dependencies
├── render.yaml                         # Render Cloud Deployment Blueprint
├── .gitignore                          # Git ignore rules
└── README.md                           # Documentation & user guide
```

---

## 🌐 Deploying to Render (Render.com)

This repository is pre-configured for automated deployment on **Render**.

### Step-by-Step Deployment Instructions

1. **Push Changes to GitHub**:
   Ensure all changes are pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure project for Render deployment"
   git push origin main
   ```

2. **Create New Web Service on Render**:
   - Log in to [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** and select **Web Service**.
   - Connect your GitHub repository: `https://github.com/psa21git/Customer-Churn-Prediction`.

3. **Configure Service Settings** (if not using `render.yaml` Blueprint):
   - **Name**: `customer-churn-prediction`
   - **Environment**: `Python 3`
   - **Region**: Select your preferred location
   - **Branch**: `main`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt && python -m src.model_training
     ```
   - **Start Command**: 
     ```bash
     gunicorn app:app
     ```

4. **Deploy**:
   - Click **Create Web Service**.
   - Render will automatically build the environment, train the model pipeline, and start the live web server!

---

## 📊 Features Used in Prediction

All 22 features identified and engineered in `customer_churn.py` are incorporated:

1. **Numerical Features**:
   - `tenure`: Customer account age in months (`StandardScaler`)
   - `MonthlyCharges`: Current monthly bill amount (`StandardScaler`)
   - `TotalCharges_yj`: Total accumulated billing amount (`PowerTransformer(method='yeo-johnson')`)

2. **Categorical Features**:
   - `SeniorCitizen`: Age 65 or older (0 / 1)
   - `Partner_Yes`: Has a partner
   - `Dependents_Yes`: Has dependents
   - `MultipleLines`: `No phone service`, `Yes` (Base: `No`)
   - `InternetService`: `Fiber optic`, `No` (Base: `DSL`)
   - `OnlineSecurity_Yes`: Online Security subscription
   - `OnlineBackup_Yes`: Online Backup subscription
   - `DeviceProtection_Yes`: Device Protection subscription
   - `TechSupport_Yes`: Tech Support subscription
   - `StreamingTV_Yes`: Streaming TV subscription
   - `StreamingMovies_Yes`: Streaming Movies subscription
   - `Contract`: `One year`, `Two year` (Base: `Month-to-month`)
   - `PaperlessBilling_Yes`: Electronic billing enabled
   - `PaymentMethod`: `Credit card (automatic)`, `Electronic check`, `Mailed check` (Base: `Bank transfer`)

*Note: Weak predictors (`gender`, `PhoneService`) and redundant `No internet service` flags are dropped to prevent overfitting and multicollinearity.*

---

## 🚀 Local Development Setup

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Model Training
```bash
python -m src.model_training
```

### 3. Run Flask Web App Locally
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000`.

---

## 🔌 API Endpoints

### 1. Single Customer Prediction
- **URL**: `/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

### 2. Batch Predictions CSV Upload
- **URL**: `/predict_batch`
- **Method**: `POST`
- **Body**: `multipart/form-data` with `file` (CSV format)

### 3. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy", "model_loaded": true}`

---

## 📈 Model Performance
- **ROC-AUC**: 0.8400
- **Overall Accuracy**: 75.51%
- **F1-Score (Churn Class)**: 0.6205
- **Churn Recall**: 75.00% (Detects 3 out of 4 potential churners via SMOTE balancing).
