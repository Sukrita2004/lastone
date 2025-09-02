# train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🚀 Starting Model Training Process for PaySim Dataset...")

try:
    # ---------------------- 1. Load Data ----------------------
    print("\nStep 1: Loading Data...")
    # Make sure your dataset file is named 'transactions.csv' or change the name here
    df = pd.read_csv('transactions.csv')
    print("✅ Data loaded successfully. Shape:", df.shape)

    # ---------------------- 2. Data Cleaning & Preprocessing ----------------------
    print("\nStep 2: Cleaning and preprocessing data...")
    
    # Drop columns that are identifiers or not useful for the model
    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    
    # One-hot encode the 'type' column
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
    print("✅ Categorical features encoded.")

    # ---------------------- 3. Feature Engineering ----------------------
    print("\nStep 3: Engineering new features...")
    # These features capture inconsistencies in balance updates, which are strong fraud indicators
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    print("✅ New features created: errorBalanceOrig, errorBalanceDest")

    # ---------------------- 4. Data Scaling and Preparation ----------------------
    print("\nStep 4: Preparing data for training...")
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    # Save the feature names and their order
    feature_names = list(X.columns)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_names)
    print("✅ Features scaled with StandardScaler.")

    # ---------------------- 5. Train/Test Split (80/20) ----------------------
    print("\nStep 5: Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Original training set fraud cases: {sum(y_train)}")

    # ---------------------- 6. Handle Imbalance with SMOTE ----------------------
    print("\nStep 6: Applying SMOTE to the training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("✅ SMOTE applied successfully.")
    print(f"New training set shape: {X_train_res.shape}")
    print(f"New training set fraud cases after SMOTE: {sum(y_train_res)}")

    # ---------------------- 7. Model Training (XGBoost) ----------------------
    print("\nStep 7: Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=8, # Deeper trees can be effective for this dataset
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='aucpr'
    )
    model.fit(X_train_res, y_train_res)
    print("✅ Model training complete.")

    # ---------------------- 8. Saving Artifacts ----------------------
    print("\nStep 8: Saving model, scaler, and feature names...")
    model.save_model("xgb_model.json")   # saves the real trees
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    print("✅ Artifacts saved successfully!")

    # ---------------------- 9. Offline Evaluation ----------------------
    print("\nStep 9: Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    print("\n--- Offline Evaluation Metrics ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print("------------------------------------\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))
    print("🎉 Model training and evaluation process finished successfully!")

except FileNotFoundError:
    print("❌ Error: 'transactions.csv' not found. Please ensure your dataset file is in the same directory.")
except Exception as e:
    print(f"❌ An error occurred: {e}")



