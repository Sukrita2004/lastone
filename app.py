# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
# --- Load Trained Artifacts ---
try:
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    print("✅ Model and artifacts loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run train_model.py first.")
    model, scaler, feature_names = None, None, None

def preprocess_input(data):
    """
    Prepares the incoming JSON data to match the model's training format.
    """
    # Create a DataFrame from the input
    df = pd.DataFrame([data])

    # 1. One-hot encode the 'type' column
    # Ensure all possible type columns exist, initialized to 0
    type_cols = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    for col in type_cols:
        df[col] = 0
    
    # Set the appropriate type column to 1
    transaction_type = data.get('type')
    if f'type_{transaction_type}' in type_cols:
        df[f'type_{transaction_type}'] = 1
    
    df = df.drop('type', axis=1)

    # 2. Engineer the same features as in training
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 3. Ensure columns are in the correct order
    # Reorder df to match the 'feature_names' list from training
    final_df = df[feature_names]

    # 4. Scale the data using the loaded scaler
    scaled_features = scaler.transform(final_df)
    final_df_scaled = pd.DataFrame(scaled_features, columns=feature_names)
    
    return final_df_scaled

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive transaction data, predict fraud, and return result."""
    if not all([model, scaler, feature_names]):
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Predict probability
        prediction_proba = model.predict_proba(processed_data)[:, 1][0]
        
        # --- Decision Logic (from Canvas Section 6) ---
        HIGH_CONFIDENCE_THRESHOLD = 0.85
        LOW_CONFIDENCE_THRESHOLD = 0.50
        
        if prediction_proba >= HIGH_CONFIDENCE_THRESHOLD:
            prediction_class = "Fraud"
            decision = "Block transaction and send notification to user/fraud team."
            confidence = "High"
        elif prediction_proba >= LOW_CONFIDENCE_THRESHOLD:
            prediction_class = "Suspicious"
            decision = "Send for manual review and monitor similar future behavior."
            confidence = "Low"
        else:
            prediction_class = "Legitimate"
            decision = "Approve transaction."
            confidence = "Low Risk"

        return jsonify({
            'prediction_class': prediction_class,
            'confidence_score': f"{prediction_proba:.4f}",
            'confidence_level': confidence,
            'recommended_action': decision
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
)


