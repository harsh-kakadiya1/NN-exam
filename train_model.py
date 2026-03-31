"""
Model Training Module for Streamlit App
Allows users to train fraud detection model on uploaded data
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Model Training Class for user-uploaded data"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_importance = None
        self.training_metrics = {}
    
    def validate_data(self, df):
        """Validate uploaded dataset"""
        required_columns = ['Time', 'Amount', 'Class']
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for V1-V28 columns (PCA features)
        v_cols = [f'V{i}' for i in range(1, 29)]
        missing_v_cols = [col for col in v_cols if col not in df.columns]
        if missing_v_cols:
            raise ValueError(f"Missing PCA features V1-V28: {missing_v_cols}")
        
        # Check if Class column has correct values (0 and 1)
        unique_classes = df['Class'].unique()
        if not all(x in [0, 1] for x in unique_classes):
            raise ValueError("Class column must contain only 0 (Normal) and 1 (Fraud)")
        
        # Check data size
        if len(df) < 100:
            raise ValueError("Dataset too small. Please upload at least 100 transactions")
        
        return True
    
    def preprocess_data(self, df):
        """Preprocess the uploaded data"""
        # Remove duplicates
        df_clean = df.drop_duplicates()
        
        # Separate features and target
        X = df_clean.drop('Class', axis=1)
        y = df_clean['Class']
        
        return X, y, df_clean
    
    def train_model(self, X, y):
        """Train the fraud detection model"""
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Handle class imbalance with SMOTE
        print("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        
        # Feature selection
        print("Selecting important features...")
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X_resampled, y_resampled)
        
        self.feature_importance = pd.DataFrame({
            'Feature': X_resampled.columns,
            'Importance': rf_temp.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Select features with importance > 0.01
        self.selected_features = self.feature_importance[
            self.feature_importance['Importance'] > 0.01
        ]['Feature'].tolist()
        
        X_selected = X_resampled[self.selected_features]
        
        # Train final model
        print("Training final model...")
        best_params = {
            'max_depth': 20,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 200,
            'random_state': 42
        }
        
        self.model = RandomForestClassifier(**best_params)
        self.model.fit(X_selected, y_resampled)
        
        # Calculate training metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_resampled, test_size=0.2, random_state=42
        )
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.training_metrics = {
            'train_accuracy': np.mean(train_pred == y_train),
            'test_accuracy': np.mean(test_pred == y_test),
            'roc_auc': roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]),
            'n_features': len(self.selected_features),
            'n_samples': len(X_resampled)
        }
        
        return self.model, self.scaler, self.selected_features
    
    def save_model(self, model_path='fraud_detection_model.pkl', 
                   scaler_path='feature_scaler.pkl', 
                   features_path='selected_features.pkl'):
        """Save trained model components"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.selected_features, features_path)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
        print(f"✅ Features saved to {features_path}")
    
    def get_training_summary(self):
        """Get training summary for display"""
        if self.model is None:
            return None
        
        summary = {
            'model_type': 'Random Forest Classifier',
            'features_used': len(self.selected_features) if self.selected_features else 0,
            'training_samples': self.training_metrics.get('n_samples', 0),
            'test_accuracy': self.training_metrics.get('test_accuracy', 0),
            'roc_auc_score': self.training_metrics.get('roc_auc', 0),
            'top_features': self.feature_importance.head(10)['Feature'].tolist() if self.feature_importance is not None else []
        }
        
        return summary

def train_from_uploaded_file(uploaded_file):
    """Train model from uploaded file"""
    
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Validate data
    trainer.validate_data(df)
    
    # Preprocess
    X, y, df_clean = trainer.preprocess_data(df)
    
    # Train model
    model, scaler, features = trainer.train_model(X, y)
    
    # Save model
    trainer.save_model()
    
    return trainer, df_clean
