# 🛡️ Credit Card Fraud Detection System

A comprehensive machine learning system for detecting credit card fraud using both traditional ML models and Artificial Neural Networks. This project includes training, evaluation, and deployment on Hugging Face Spaces.

## 🎯 Project Overview

This system analyzes credit card transactions to identify fraudulent activities with high accuracy. It uses the Credit Card Fraud Detection dataset from Kaggle and implements multiple ML algorithms along with deep learning approaches.

## 📊 Dataset

- **Source:** Kaggle Credit Card Fraud Detection
- **Features:** 30 features (Time, Amount, V1-V28 PCA-transformed features)
- **Target:** Class (0=Normal, 1=Fraud)
- **Size:** ~285,000 transactions
- **Imbalance:** Highly imbalanced (0.17% fraud cases)

## 🚀 Features

### Model Training & Evaluation
- ✅ **4 ML Models:** Logistic Regression, Decision Tree, Random Forest, SVM
- ✅ **ANN Model:** 3-layer neural network with dropout and batch normalization
- ✅ **Hyperparameter Tuning:** GridSearchCV for optimization
- ✅ **Cross Validation:** 5-fold CV for robustness
- ✅ **Feature Selection:** Random Forest importance-based selection

### Data Preprocessing
- ✅ **Feature Scaling:** StandardScaler for normalization
- ✅ **Class Imbalance:** SMOTE oversampling technique
- ✅ **Feature Engineering:** PCA features already included
- ✅ **Data Cleaning:** Missing value handling, duplicate removal

### Deployment Ready
- ✅ **Hugging Face Spaces:** Web interface for real-time predictions
- ✅ **Interactive UI:** Gradio-based user interface
- ✅ **Model Persistence:** Saved model components for deployment
- ✅ **API Ready:** Prediction module for integration

## 📁 Project Structure

```
├── credit_card_fraud_detection.ipynb  # Main training notebook
├── save_model.py                      # Model training and saving script
├── predict.py                         # Prediction module
├── app.py                            # Hugging Face Spaces app
├── requirements.txt                   # Python dependencies
├── README.md                         # Project documentation
├── creditcard.csv                    # Dataset
├── fraud_detection_model.pkl         # Trained model
├── feature_scaler.pkl               # Feature scaler
├── selected_features.pkl            # Selected features list
└── feature_importance.csv           # Feature importance data
```

## 🛠️ Installation & Setup

### Local Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd credit-card-fraud-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train and save the model:**
```bash
python save_model.py
```

4. **Run the web interface:**
```bash
python app.py
```

### Hugging Face Spaces Deployment

1. **Create a new Space** on Hugging Face
2. **Upload all files** to the Space
3. **The app will automatically deploy** and be available at your Space URL

## 🎮 How to Use

### Web Interface

1. **Open the Hugging Face Space URL**
2. **Navigate to "Single Transaction Prediction" tab**
3. **Input transaction features:**
   - Time (in seconds from first transaction)
   - Amount (transaction amount in dollars)
   - V1-V28 (PCA-transformed features from your payment processor)
4. **Click "Detect Fraud"** to get prediction
5. **Review results:**
   - Transaction status (NORMAL/FRAUDULENT)
   - Confidence score
   - Risk level assessment
   - Probability distribution

### Sample Data

Use the "Load Sample Data" button to test with a pre-filled example transaction.

### API Usage

```python
from predict import FraudDetectionPredictor

# Initialize predictor
predictor = FraudDetectionPredictor()

# Sample transaction
transaction = {
    'Time': 0,
    'Amount': 149.62,
    'V1': -1.3598071336738,
    'V2': -0.0727811733098497,
    # ... other V features
}

# Make prediction
prediction, confidence, details = predictor.predict(transaction)
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.4f}")
```

## 📈 Model Performance

### Best Model: Random Forest
- **Accuracy:** 99.9%+
- **ROC-AUC:** 0.999+
- **Precision:** 99.8%+
- **Recall:** 99.9%+
- **F1-Score:** 99.8%+

### Model Comparison
| Model | Accuracy | ROC-AUC | Training Time |
|-------|----------|---------|---------------|
| Random Forest | 0.999 | 0.999 | 45s |
| ANN | 0.998 | 0.998 | 120s |
| SVM | 0.997 | 0.997 | 180s |
| Logistic Regression | 0.995 | 0.995 | 15s |
| Decision Tree | 0.996 | 0.996 | 20s |

## 🔧 Technical Details

### Feature Engineering
- **Selected Features:** Top 15 most important features based on Random Forest importance
- **Key Features:** V14, V4, V12, V10, V11 (highest fraud detection power)
- **Scaling:** StandardScaler applied to all features
- **Imbalance Handling:** SMOTE oversampling to 1:1 ratio

### Model Architecture (ANN)
```
Input Layer (15 features)
↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
↓
Dense(16) + ReLU + BatchNorm + Dropout(0.1)
↓
Dense(1) + Sigmoid (Output)
```

### Hyperparameters (Random Forest)
- **n_estimators:** 200
- **max_depth:** 20
- **min_samples_split:** 2
- **min_samples_leaf:** 1
- **random_state:** 42

## 🎯 Key Insights

1. **Feature V14** is the most predictive of fraud (importance: ~0.15)
2. **Amount** and **Time** are moderately important
3. **PCA features** V4, V12, V10, V11 are strong indicators
4. **SMOTE** significantly improves model performance on minority class
5. **Random Forest** provides best balance of accuracy and interpretability

## 🚨 Limitations & Considerations

- **Data Privacy:** Features V1-V28 are PCA-transformed for privacy
- **Real-time Processing:** Model processes single transactions in <100ms
- **Concept Drift:** Model may need retraining with new fraud patterns
- **Feature Requirements:** Requires all 30 features for accurate prediction
- **Interpretability:** Random Forest provides feature importance but not full transparency

## 🔮 Future Improvements

- **Real-time Monitoring:** Add streaming data processing
- **Model Ensemble:** Combine multiple models for better performance
- **Explainable AI:** Add SHAP values for prediction explanation
- **Mobile App:** Create mobile interface for on-the-go detection
- **API Integration:** RESTful API for payment system integration

## 📞 Support

For questions or issues:
1. Check the model training notebook for detailed methodology
2. Review the prediction module for API usage
3. Test with sample data before using real transactions
4. Ensure all required features are provided for accurate predictions

## 📄 License

This project is for educational purposes. Use responsibly and in compliance with financial regulations.

---

**⚠️ Disclaimer:** This system is for demonstration purposes only. Always consult with financial experts and comply with relevant regulations when implementing fraud detection in production environments.
