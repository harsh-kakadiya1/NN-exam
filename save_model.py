"""
Model Saving Script for Hugging Face Deployment
Saves the trained fraud detection model and all necessary components
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Train the model and save all components for deployment"""
    
    print("🔄 Loading and preprocessing data...")
    
    # Load dataset
    df = pd.read_csv('creditcard.csv')
    df = df.drop_duplicates()
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Feature selection
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_resampled, y_resampled)
    
    feature_importance = pd.DataFrame({
        'Feature': X_resampled.columns,
        'Importance': rf_temp.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    selected_features = feature_importance[feature_importance['Importance'] > 0.01]['Feature'].tolist()
    X_selected = X_resampled[selected_features]
    
    # Train final model with best parameters
    print("🎯 Training final model...")
    best_params = {
        'max_depth': 20,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 200
    }
    
    final_model = RandomForestClassifier(random_state=42, **best_params)
    final_model.fit(X_selected, y_resampled)
    
    # Save all components
    print("💾 Saving model components...")
    joblib.dump(final_model, 'fraud_detection_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')
    
    # Save feature importance for reference
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print("✅ Model and components saved successfully!")
    print(f"📊 Selected {len(selected_features)} features: {selected_features}")
    
    # Test the saved model
    print("🧪 Testing saved model...")
    test_model = joblib.load('fraud_detection_model.pkl')
    test_scaler = joblib.load('feature_scaler.pkl')
    test_features = joblib.load('selected_features.pkl')
    
    # Create a sample test
    sample_data = X_selected.iloc[:5]
    predictions = test_model.predict(sample_data)
    probabilities = test_model.predict_proba(sample_data)
    
    print("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Sample {i+1}: {'FRAUD' if pred == 1 else 'NORMAL'} (Confidence: {max(prob):.4f})")
    
    return final_model, scaler, selected_features

if __name__ == "__main__":
    model, scaler, features = train_and_save_model()
