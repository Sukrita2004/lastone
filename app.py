from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

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
    df = pd.DataFrame([data])

    # One-hot encode the 'type' column
    type_cols = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    for col in type_cols:
        df[col] = 0
    
    transaction_type = data.get('type')
    if f'type_{transaction_type}' in type_cols:
        df[f'type_{transaction_type}'] = 1
    
    df = df.drop('type', axis=1)

    # Feature engineering
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Ensure columns match training
    final_df = df[feature_names]

    # Scale features
    scaled_features = scaler.transform(final_df)
    final_df_scaled = pd.DataFrame(scaled_features, columns=feature_names)
    
    return final_df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, scaler, feature_names]):
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Predict class and probabilities
        prediction = model.predict(processed_data)[0]
        proba = model.predict_proba(processed_data)[0]

        fraud_proba = proba[1]
        legit_proba = proba[0]

        # Assign label and confidence based on prediction
        if prediction == 1:  # Fraud
            label = "Fraud"
            confidence = fraud_proba
            recommended_action = "Block transaction and send notification to user/fraud team."
        else:  # Legitimate
            label = "Legitimate"
            confidence = legit_proba
            recommended_action = "Approve transaction."

        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "High Risk" if label == "Fraud" else "High Confidence"
        elif confidence >= 0.5:
            confidence_level = "Medium Risk" if label == "Fraud" else "Moderate Confidence"
        else:
            confidence_level = "Low Risk" if label == "Fraud" else "Low Confidence"

        return jsonify({
            'prediction_class': label,
            'confidence_score': f"{confidence:.4f}",
            'confidence_level': confidence_level,
            'recommended_action': recommended_action
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
