import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib

FEATURE_NAMES = [
    'tenure',
    'MonthlyCharges',
    'SeniorCitizen',
    'Partner_Yes',
    'Dependents_Yes',
    'MultipleLines_No phone service',
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'TotalCharges_yj'
]


def _safe_float(val, default=0.0):
    """Safely converts string/number/NaN value to float."""
    if pd.isna(val) or val is None:
        return default
    try:
        val_str = str(val).strip()
        if val_str == "" or val_str.lower() == "nan":
            return default
        return float(val_str)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Safely converts string/number/boolean value to int (0 or 1)."""
    if pd.isna(val) or val is None:
        return default
    if isinstance(val, bool):
        return 1 if val else 0
    val_str = str(val).strip().lower()
    if val_str in ['1', 'yes', 'true']:
        return 1
    return default


def _safe_yes_no(val):
    """Normalizes string/bool inputs into 'Yes' or 'No'."""
    if isinstance(val, bool):
        return 'Yes' if val else 'No'
    val_str = str(val).strip().lower()
    if val_str in ['yes', '1', 'true', 'on']:
        return 'Yes'
    return 'No'


class TelcoDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.is_fitted = False

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans raw dataset by removing identifiers/weak features, handling data types,
        imputing missing values, and performing one-hot encoding matching customer_churn.py.
        """
        df_clean = df.copy()

        # Drop identifiers if present
        if 'customerID' in df_clean.columns:
            df_clean.drop(columns=['customerID'], inplace=True)

        # Convert TotalCharges to numeric & coerce errors
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            median_val = df_clean['TotalCharges'].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_clean['TotalCharges'].fillna(median_val, inplace=True)

        # Remove weak predictors as in customer_churn.py
        for col in ['gender', 'PhoneService']:
            if col in df_clean.columns:
                df_clean.drop(columns=[col], inplace=True)

        return df_clean

    def preprocess_train_data(self, df: pd.DataFrame):
        """
        Preprocesses full dataset for training, fits scalers/transformers, and returns X and y.
        """
        df_clean = self.clean_raw_data(df)

        cat_cols = [
            'SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]

        if 'Churn' in df_clean.columns:
            cat_cols.append('Churn')

        cols_to_encode = [c for c in cat_cols if c in df_clean.columns]

        # One-Hot Encoding
        encoded_df = pd.get_dummies(df_clean[cols_to_encode], drop_first=True).astype(int)
        df_clean = pd.concat([df_clean.drop(columns=cols_to_encode), encoded_df], axis=1)

        # Drop redundant 'No internet service' columns as done in customer_churn.py
        cols_to_be_dropped = [
            'OnlineSecurity_No internet service',
            'OnlineBackup_No internet service',
            'DeviceProtection_No internet service',
            'TechSupport_No internet service',
            'StreamingTV_No internet service',
            'StreamingMovies_No internet service'
        ]
        df_clean.drop(columns=[c for c in cols_to_be_dropped if c in df_clean.columns], inplace=True, errors='ignore')

        # Fit and Transform Scaler for numerical features (tenure, MonthlyCharges)
        num_cols = ['tenure', 'MonthlyCharges']
        df_clean[num_cols] = self.scaler.fit_transform(df_clean[num_cols])

        # Fit and Transform PowerTransformer for TotalCharges (Yeo-Johnson)
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges_yj'] = self.power_transformer.fit_transform(df_clean[['TotalCharges']])
            df_clean.drop(columns=['TotalCharges'], inplace=True)

        self.is_fitted = True

        # Extract target if present
        if 'Churn_Yes' in df_clean.columns:
            X = df_clean.drop(columns=['Churn_Yes'])
            y = df_clean['Churn_Yes']
        else:
            X = df_clean
            y = None

        # Reorder/ensure X matches FEATURE_NAMES
        for col in FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0

        X = X[FEATURE_NAMES]
        return X, y

    def transform_single_input(self, input_dict: dict) -> pd.DataFrame:
        """
        Transforms a dictionary of raw customer attributes into a single-row feature DataFrame
        compatible with the trained ML model. Robustly handles string, numeric, and boolean types.
        """
        senior = _safe_int(input_dict.get('SeniorCitizen', 0))
        partner = _safe_yes_no(input_dict.get('Partner', 'No'))
        dependents = _safe_yes_no(input_dict.get('Dependents', 'No'))
        tenure = _safe_float(input_dict.get('tenure', 0.0))
        multiple_lines = str(input_dict.get('MultipleLines', 'No')).strip()
        internet = str(input_dict.get('InternetService', 'No')).strip()
        sec = _safe_yes_no(input_dict.get('OnlineSecurity', 'No'))
        backup = _safe_yes_no(input_dict.get('OnlineBackup', 'No'))
        protection = _safe_yes_no(input_dict.get('DeviceProtection', 'No'))
        tech = _safe_yes_no(input_dict.get('TechSupport', 'No'))
        tv = _safe_yes_no(input_dict.get('StreamingTV', 'No'))
        movies = _safe_yes_no(input_dict.get('StreamingMovies', 'No'))
        contract = str(input_dict.get('Contract', 'Month-to-month')).strip()
        paperless = _safe_yes_no(input_dict.get('PaperlessBilling', 'No'))
        payment = str(input_dict.get('PaymentMethod', 'Electronic check')).strip()
        monthly = _safe_float(input_dict.get('MonthlyCharges', 0.0))
        total = _safe_float(input_dict.get('TotalCharges', 0.0))

        # Scale tenure and MonthlyCharges
        num_df = pd.DataFrame([{'tenure': tenure, 'MonthlyCharges': monthly}])
        scaled_num = self.scaler.transform(num_df)

        # Transform TotalCharges with Yeo-Johnson
        tot_df = pd.DataFrame([{'TotalCharges': total}])
        tot_yj = self.power_transformer.transform(tot_df)[0][0]

        features = {}
        features['tenure'] = float(scaled_num[0][0])
        features['MonthlyCharges'] = float(scaled_num[0][1])
        features['TotalCharges_yj'] = float(tot_yj)

        features['SeniorCitizen'] = senior
        features['Partner_Yes'] = 1 if partner == 'Yes' else 0
        features['Dependents_Yes'] = 1 if dependents == 'Yes' else 0

        features['MultipleLines_No phone service'] = 1 if multiple_lines.lower() == 'no phone service' else 0
        features['MultipleLines_Yes'] = 1 if multiple_lines.lower() in ['yes', 'true', '1'] else 0

        features['InternetService_Fiber optic'] = 1 if internet.lower() in ['fiber optic', 'fiber'] else 0
        features['InternetService_No'] = 1 if internet.lower() in ['no', 'none'] else 0

        features['OnlineSecurity_Yes'] = 1 if sec == 'Yes' else 0
        features['OnlineBackup_Yes'] = 1 if backup == 'Yes' else 0
        features['DeviceProtection_Yes'] = 1 if protection == 'Yes' else 0
        features['TechSupport_Yes'] = 1 if tech == 'Yes' else 0
        features['StreamingTV_Yes'] = 1 if tv == 'Yes' else 0
        features['StreamingMovies_Yes'] = 1 if movies == 'Yes' else 0

        features['Contract_One year'] = 1 if contract.lower() in ['one year', '1 year'] else 0
        features['Contract_Two year'] = 1 if contract.lower() in ['two year', '2 year'] else 0

        features['PaperlessBilling_Yes'] = 1 if paperless == 'Yes' else 0

        features['PaymentMethod_Credit card (automatic)'] = 1 if 'credit card' in payment.lower() else 0
        features['PaymentMethod_Electronic check'] = 1 if 'electronic check' in payment.lower() else 0
        features['PaymentMethod_Mailed check'] = 1 if 'mailed check' in payment.lower() else 0

        feature_df = pd.DataFrame([features])[FEATURE_NAMES]
        return feature_df

    def save_artifacts(self, artifact_dir: str):
        """Saves fitted scaler and power transformer to artifact directory."""
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(artifact_dir, 'scaler.pkl'))
        joblib.dump(self.power_transformer, os.path.join(artifact_dir, 'power_transformer.pkl'))

    def load_artifacts(self, artifact_dir: str):
        """Loads scaler and power transformer from artifact directory."""
        self.scaler = joblib.load(os.path.join(artifact_dir, 'scaler.pkl'))
        self.power_transformer = joblib.load(os.path.join(artifact_dir, 'power_transformer.pkl'))
        self.is_fitted = True
