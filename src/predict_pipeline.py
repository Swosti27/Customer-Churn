import os
import json
import pandas as pd
import numpy as np
import joblib

from src.data_preprocessing import TelcoDataPreprocessor, FEATURE_NAMES, _safe_float


class PredictPipeline:
    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = artifact_dir
        self.model = None
        self.preprocessor = TelcoDataPreprocessor()
        self.feature_names = FEATURE_NAMES
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.artifact_dir, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train the model first.")

        self.model = joblib.load(model_path)
        self.preprocessor.load_artifacts(self.artifact_dir)

        features_path = os.path.join(self.artifact_dir, "feature_names.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)

    def predict_single(self, input_dict: dict) -> dict:
        """
        Predicts churn risk and probability for a single customer raw attribute dict.
        """
        # Transform input into model-ready DataFrame
        feature_df = self.preprocessor.transform_single_input(input_dict)

        # Predict probability and class
        prob = float(self.model.predict_proba(feature_df)[0][1])
        prediction = int(prob >= 0.5)

        if prob >= 0.65:
            risk_level = "HIGH"
            risk_color = "#dc2626"  # Red
        elif prob >= 0.35:
            risk_level = "MEDIUM"
            risk_color = "#d97706"  # Amber
        else:
            risk_level = "LOW"
            risk_color = "#16a34a"  # Green

        label = "Churn (At Risk)" if prediction == 1 else "Retain (Loyal)"

        # Generate custom retention recommendations
        recommendations = self._generate_recommendations(input_dict, prob)

        return {
            "prediction": prediction,
            "prediction_label": label,
            "churn_probability": round(prob * 100, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "recommendations": recommendations
        }

    def predict_batch(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Processes a batch DataFrame of customer records. Automatically handles raw datasets (Telco-Customer-Churn.csv),
        pre-cleaned datasets (cleaned_dataset.csv), or hybrid customer datasets.
        """
        df_out = df.copy()

        # Remove index/unnamed columns if present
        cols_to_drop = [c for c in ['Unnamed: 0', ''] if c in df_out.columns]
        if cols_to_drop:
            df_out.drop(columns=cols_to_drop, inplace=True)

        batch_features = self._transform_batch_df(df_out)

        probs = self.model.predict_proba(batch_features)[:, 1]

        df_out['Churn_Probability_%'] = np.round(probs * 100, 2)
        df_out['Predicted_Churn'] = (probs >= 0.5).astype(int)
        df_out['Predicted_Label'] = np.where(probs >= 0.5, 'Churn', 'No Churn')
        df_out['Risk_Level'] = np.where(probs >= 0.65, 'HIGH', np.where(probs >= 0.35, 'MEDIUM', 'LOW'))

        summary = {
            "total_records": len(df_out),
            "predicted_churn": int((probs >= 0.5).sum()),
            "predicted_retain": int((probs < 0.5).sum()),
            "high_risk_count": int((probs >= 0.65).sum()),
            "medium_risk_count": int(((probs >= 0.35) & (probs < 0.65)).sum()),
            "low_risk_count": int((probs < 0.35).sum())
        }

        return df_out, summary

    def _transform_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely converts any input DataFrame (raw, pre-cleaned, or hybrid) into the 22 feature_names vector.
        """
        # If all 22 feature names are already present in df (pre-cleaned format)
        has_all_features = all(col in df.columns for col in self.feature_names)
        if has_all_features:
            batch_df = df[self.feature_names].copy()
            batch_df = batch_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            return batch_df

        processed_rows = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()

            # Baseline transformation from raw input dict (handles scaling tenure, MonthlyCharges, TotalCharges_yj)
            feat_df = self.preprocessor.transform_single_input(row_dict)

            # Override categorical encoded columns if directly present in dataset
            for col in self.feature_names:
                if col not in ['tenure', 'MonthlyCharges', 'TotalCharges_yj']:
                    if col in row_dict and not pd.isna(row_dict[col]):
                        try:
                            val = float(row_dict[col])
                            feat_df[col] = val
                        except (ValueError, TypeError):
                            pass

            processed_rows.append(feat_df)

        batch_features = pd.concat(processed_rows, ignore_index=True)
        return batch_features

    def _generate_recommendations(self, input_dict: dict, churn_prob: float) -> list:
        recs = []

        contract = str(input_dict.get('Contract', 'Month-to-month')).strip()
        tenure = _safe_float(input_dict.get('tenure', 0))
        tech_support = str(input_dict.get('TechSupport', 'No')).strip().lower()
        online_sec = str(input_dict.get('OnlineSecurity', 'No')).strip().lower()
        payment = str(input_dict.get('PaymentMethod', 'Electronic check')).strip().lower()
        internet = str(input_dict.get('InternetService', 'No')).strip().lower()

        if contract == 'Month-to-month' or contract == '1':
            recs.append("Offer a discount on 1-year or 2-year annual contract to improve customer commitment.")

        if tenure <= 12 and churn_prob >= 0.4:
            recs.append("New customer in critical onboarding window: Send targeted engagement & loyalty rewards.")

        if tech_support not in ['yes', '1', 'true'] and internet not in ['no', 'none']:
            recs.append("Bundle complimentary Tech Support for 3 months to improve service satisfaction.")

        if online_sec not in ['yes', '1', 'true'] and internet not in ['no', 'none']:
            recs.append("Promote Online Security add-on trial to increase product stickiness.")

        if 'electronic check' in payment:
            recs.append("Encourage switching to Automatic Credit Card or Bank Transfer with a small billing credit.")

        if len(recs) == 0:
            recs.append("Customer shows strong loyalty signals. Maintain high service standard and cross-sell relevant features.")

        return recs
