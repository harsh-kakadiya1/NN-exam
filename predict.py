"""
Fraud Detection Prediction Module
Handles single transaction prediction for user input
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple

class FraudDetectionPredictor:
    """Fraud Detection Predictor Class"""
    
    def __init__(self, model_path: str = 'fraud_detection_model.pkl',
                 scaler_path: str = 'feature_scaler.pkl',
                 features_path: str = 'selected_features.pkl'):
        """
        Initialize the predictor with saved model components
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            features_path: Path to selected features list
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.selected_features = joblib.load(features_path)
            print("✅ Model components loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def preprocess_input(self, transaction_data: Dict) -> pd.DataFrame:
        """
        Preprocess user input transaction data
        
        Args:
            transaction_data: Dictionary containing transaction features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([transaction_data])
        
        # Ensure all required columns are present
        required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                           'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
                           'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 
                           'V25', 'V26', 'V27', 'V28', 'Amount']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Scale the features
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=required_columns)
        
        # Select only the features used by the model
        df_selected = df_scaled[self.selected_features]
        
        return df_selected
    
    def predict(self, transaction_data: Dict) -> Tuple[str, float, Dict]:
        """
        Make prediction on transaction data
        
        Args:
            transaction_data: Dictionary containing transaction features
            
        Returns:
            Tuple of (prediction, confidence, detailed_results)
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(transaction_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probabilities = self.model.predict_proba(processed_data)[0]
            
            # Get confidence
            confidence = max(probabilities)
            
            # Prepare results
            result = {
                'prediction': 'FRAUDULENT' if prediction == 1 else 'NORMAL',
                'confidence': confidence,
                'fraud_probability': probabilities[1],
                'normal_probability': probabilities[0],
                'risk_level': self._get_risk_level(probabilities[1]),
                'features_used': self.selected_features,
                'raw_prediction': int(prediction)
            }
            
            return result['prediction'], confidence, result
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return "ERROR", 0.0, {'error': str(e)}
    
    def _get_risk_level(self, fraud_probability: float) -> str:
        """Determine risk level based on fraud probability"""
        if fraud_probability >= 0.8:
            return "HIGH RISK"
        elif fraud_probability >= 0.5:
            return "MEDIUM RISK"
        elif fraud_probability >= 0.2:
            return "LOW RISK"
        else:
            return "VERY LOW RISK"
    
    def batch_predict(self, transactions_list: list) -> list:
        """
        Make predictions on multiple transactions
        
        Args:
            transactions_list: List of transaction dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for transaction in transactions_list:
            _, _, result = self.predict(transaction)
            results.append(result)
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.selected_features,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance_df
        else:
            return pd.DataFrame(columns=['Feature', 'Importance'])

# Example usage function
def example_prediction():
    """Example of how to use the predictor"""
    
    # Initialize predictor
    predictor = FraudDetectionPredictor()
    
    # Sample transaction data (you would get this from user input)
    sample_transaction = {
        'Time': 0,
        'V1': -1.3598071336738,
        'V2': -0.0727811733098497,
        'V3': 2.53634673796914,
        'V4': 1.37815522427443,
        'V5': -0.338320769942518,
        'V6': 0.462387777762292,
        'V7': 0.239598554061257,
        'V8': 0.0986979012610507,
        'V9': 0.363786969611213,
        'V10': 0.0907941719789316,
        'V11': -0.551599533260813,
        'V12': -0.617800855762348,
        'V13': -0.991389847235408,
        'V14': -0.311169353699879,
        'V15': 1.46817697209427,
        'V16': -0.470400525259478,
        'V17': 0.207971241929242,
        'V18': 0.0257905801985591,
        'V19': 0.403992960255733,
        'V20': 0.251412098239705,
        'V21': -0.018306777944153,
        'V22': 0.277837575558899,
        'V23': -0.110473910188767,
        'V24': 0.0669280749146731,
        'V25': 0.128539358273528,
        'V26': -0.189114843888824,
        'V27': 0.133558376740387,
        'V28': -0.0210530534538215,
        'Amount': 149.62
    }
    
    # Make prediction
    prediction, confidence, details = predictor.predict(sample_transaction)
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Fraud Probability: {details['fraud_probability']:.4f}")
    print(f"Risk Level: {details['risk_level']}")

if __name__ == "__main__":
    example_prediction()
