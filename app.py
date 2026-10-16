import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from src.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Global predictor pipeline instance
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = PredictPipeline(artifact_dir="artifacts")
    return pipeline


@app.route("/")
def index():
    """Renders the main Customer Churn Prediction dashboard web page."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify server and model status."""
    try:
        predictor = get_pipeline()
        is_loaded = predictor.model is not None
        return jsonify({"status": "healthy", "model_loaded": is_loaded}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for single customer churn prediction.
    Supports both JSON payloads and HTML form submissions.
    """
    try:
        if request.is_json:
            input_data = request.get_json()
        else:
            input_data = request.form.to_dict()

        predictor = get_pipeline()
        result = predictor.predict_single(input_data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Endpoint for batch prediction via uploaded CSV file.
    Returns JSON containing summary statistics, preview rows, and CSV download link.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded. Please select a valid CSV file."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File format must be CSV (.csv)."}), 400

        df = pd.read_csv(file)
        predictor = get_pipeline()
        result_df, summary = predictor.predict_batch(df)

        output_path = os.path.join("artifacts", "batch_predictions_output.csv")
        os.makedirs("artifacts", exist_ok=True)
        result_df.to_csv(output_path, index=False)

        # Sanitize NaN/Inf for JSON response
        result_df_clean = result_df.fillna("N/A")
        preview_rows = result_df_clean.head(20).to_dict(orient="records")

        return jsonify({
            "success": True,
            "summary": summary,
            "preview": preview_rows,
            "columns": list(result_df.columns),
            "download_url": "/download_batch"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_batch", methods=["GET"])
def download_batch():
    """Downloads the generated batch predictions CSV file."""
    output_path = os.path.join("artifacts", "batch_predictions_output.csv")
    if not os.path.exists(output_path):
        return jsonify({"error": "No batch prediction file found."}), 404

    return send_file(
        output_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="churn_predictions.csv"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
