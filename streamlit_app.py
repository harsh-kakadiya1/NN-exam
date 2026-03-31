"""
Credit Card Fraud Detection Streamlit Application
A web interface for users to input transaction data and get fraud predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from predict import FraudDetectionPredictor
from train_model import ModelTrainer, train_from_uploaded_file
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff8800;
        font-weight: bold;
    }
    .risk-low {
        color: #00C851;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the predictor
@st.cache_resource
def load_predictor():
    """Load the fraud detection model"""
    try:
        predictor = FraudDetectionPredictor()
        return predictor, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

# Load the model
predictor, model_loaded = load_predictor()

def create_input_fields():
    """Create input fields for all transaction features"""
    
    st.markdown("### 📝 Transaction Details")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        time_input = st.number_input("⏰ Time (seconds)", value=0, step=1, help="Time elapsed since first transaction")
        amount_input = st.number_input("💰 Amount ($)", value=100.0, step=0.01, help="Transaction amount")
        
        st.markdown("#### PCA Features (V1-V14)")
        v1_input = st.number_input("V1", value=0.0, step=0.0001, format="%.6f")
        v2_input = st.number_input("V2", value=0.0, step=0.0001, format="%.6f")
        v3_input = st.number_input("V3", value=0.0, step=0.0001, format="%.6f")
        v4_input = st.number_input("V4", value=0.0, step=0.0001, format="%.6f")
        v5_input = st.number_input("V5", value=0.0, step=0.0001, format="%.6f")
        v6_input = st.number_input("V6", value=0.0, step=0.0001, format="%.6f")
        v7_input = st.number_input("V7", value=0.0, step=0.0001, format="%.6f")
        v8_input = st.number_input("V8", value=0.0, step=0.0001, format="%.6f")
    
    with col2:
        st.markdown("#### PCA Features (V9-V28)")
        v9_input = st.number_input("V9", value=0.0, step=0.0001, format="%.6f")
        v10_input = st.number_input("V10", value=0.0, step=0.0001, format="%.6f")
        v11_input = st.number_input("V11", value=0.0, step=0.0001, format="%.6f")
        v12_input = st.number_input("V12", value=0.0, step=0.0001, format="%.6f")
        v13_input = st.number_input("V13", value=0.0, step=0.0001, format="%.6f")
        v14_input = st.number_input("V14", value=0.0, step=0.0001, format="%.6f")
        v15_input = st.number_input("V15", value=0.0, step=0.0001, format="%.6f")
        v16_input = st.number_input("V16", value=0.0, step=0.0001, format="%.6f")
        v17_input = st.number_input("V17", value=0.0, step=0.0001, format="%.6f")
        v18_input = st.number_input("V18", value=0.0, step=0.0001, format="%.6f")
        v19_input = st.number_input("V19", value=0.0, step=0.0001, format="%.6f")
        v20_input = st.number_input("V20", value=0.0, step=0.0001, format="%.6f")
        v21_input = st.number_input("V21", value=0.0, step=0.0001, format="%.6f")
        v22_input = st.number_input("V22", value=0.0, step=0.0001, format="%.6f")
        v23_input = st.number_input("V23", value=0.0, step=0.0001, format="%.6f")
        v24_input = st.number_input("V24", value=0.0, step=0.0001, format="%.6f")
        v25_input = st.number_input("V25", value=0.0, step=0.0001, format="%.6f")
        v26_input = st.number_input("V26", value=0.0, step=0.0001, format="%.6f")
        v27_input = st.number_input("V27", value=0.0, step=0.0001, format="%.6f")
        v28_input = st.number_input("V28", value=0.0, step=0.0001, format="%.6f")
    
    return {
        'time_input': time_input, 'amount_input': amount_input,
        'v1_input': v1_input, 'v2_input': v2_input, 'v3_input': v3_input, 'v4_input': v4_input,
        'v5_input': v5_input, 'v6_input': v6_input, 'v7_input': v7_input, 'v8_input': v8_input,
        'v9_input': v9_input, 'v10_input': v10_input, 'v11_input': v11_input, 'v12_input': v12_input,
        'v13_input': v13_input, 'v14_input': v14_input, 'v15_input': v15_input, 'v16_input': v16_input,
        'v17_input': v17_input, 'v18_input': v18_input, 'v19_input': v19_input, 'v20_input': v20_input,
        'v21_input': v21_input, 'v22_input': v22_input, 'v23_input': v23_input, 'v24_input': v24_input,
        'v25_input': v25_input, 'v26_input': v26_input, 'v27_input': v27_input, 'v28_input': v28_input
    }

def predict_fraud(input_data):
    """Make fraud prediction based on user input"""
    
    if not model_loaded:
        return None, None, None
    
    try:
        # Create transaction dictionary
        transaction_data = {
            'Time': input_data['time_input'],
            'Amount': input_data['amount_input'],
            'V1': input_data['v1_input'], 'V2': input_data['v2_input'], 'V3': input_data['v3_input'], 'V4': input_data['v4_input'],
            'V5': input_data['v5_input'], 'V6': input_data['v6_input'], 'V7': input_data['v7_input'], 'V8': input_data['v8_input'],
            'V9': input_data['v9_input'], 'V10': input_data['v10_input'], 'V11': input_data['v11_input'], 'V12': input_data['v12_input'],
            'V13': input_data['v13_input'], 'V14': input_data['v14_input'], 'V15': input_data['v15_input'], 'V16': input_data['v16_input'],
            'V17': input_data['v17_input'], 'V18': input_data['v18_input'], 'V19': input_data['v19_input'], 'V20': input_data['v20_input'],
            'V21': input_data['v21_input'], 'V22': input_data['v22_input'], 'V23': input_data['v23_input'], 'V24': input_data['v24_input'],
            'V25': input_data['v25_input'], 'V26': input_data['v26_input'], 'V27': input_data['v27_input'], 'V28': input_data['v28_input']
        }
        
        # Make prediction
        prediction, confidence, details = predictor.predict(transaction_data)
        
        return prediction, confidence, details
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def create_probability_chart(details):
    """Create probability distribution chart"""
    fig = go.Figure(data=[
        go.Bar(name='Normal', x=['Probability'], y=[details['normal_probability']], 
               marker_color='green', text=[f"{details['normal_probability']:.3f}"], textposition='auto'),
        go.Bar(name='Fraud', x=['Probability'], y=[details['fraud_probability']], 
               marker_color='red', text=[f"{details['fraud_probability']:.3f}"], textposition='auto')
    ])
    fig.update_layout(
        title="Transaction Probability Distribution",
        yaxis_title="Probability",
        barmode='group',
        height=400,
        showlegend=True
    )
    return fig

def create_risk_gauge(details):
    """Create risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = details['fraud_probability'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 50], 'color': "yellow"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=400)
    return fig

def load_sample_data():
    """Load sample transaction data for testing"""
    return {
        'time_input': 0,
        'amount_input': 149.62,
        'v1_input': -1.3598071336738,
        'v2_input': -0.0727811733098497,
        'v3_input': 2.53634673796914,
        'v4_input': 1.37815522427443,
        'v5_input': -0.338320769942518,
        'v6_input': 0.462387777762292,
        'v7_input': 0.239598554061257,
        'v8_input': 0.0986979012610507,
        'v9_input': 0.363786969611213,
        'v10_input': 0.0907941719789316,
        'v11_input': -0.551599533260813,
        'v12_input': -0.617800855762348,
        'v13_input': -0.991389847235408,
        'v14_input': -0.311169353699879,
        'v15_input': 1.46817697209427,
        'v16_input': -0.470400525259478,
        'v17_input': 0.207971241929242,
        'v18_input': 0.0257905801985591,
        'v19_input': 0.403992960255733,
        'v20_input': 0.251412098239705,
        'v21_input': -0.018306777944153,
        'v22_input': 0.277837575558899,
        'v23_input': -0.110473910188767,
        'v24_input': 0.0669280749146731,
        'v25_input': 0.128539358273528,
        'v26_input': -0.189114843888824,
        'v27_input': 0.133558376740387,
        'v28_input': -0.0210530534538215
    }

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Model status
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ No model loaded. Upload data and train a model in the 'Train Model' section first.")
        # Don't return here - allow user to navigate to training page
    
    # Sidebar with navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["🔍 Fraud Detection", "📊 Model Information", "🎯 Train Model", "📈 Feature Importance"])
    
    if page == "🔍 Fraud Detection":
        # Main prediction interface
        st.markdown("---")
        
        if not model_loaded:
            st.warning("🚫 No model available for predictions. Please train a model first in the 'Train Model' section.")
            st.info("👈 Go to 'Train Model' in the sidebar to upload your data and train a fraud detection model.")
        else:
            # Input fields
            input_fields = create_input_fields()
            
            # Buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                predict_btn = st.button("🔍 Detect Fraud", type="primary", use_container_width=True)
            with col2:
                sample_btn = st.button("📋 Load Sample Data", use_container_width=True)
            
            # Load sample data if button clicked
            if sample_btn:
                sample_data = load_sample_data()
                for key, value in sample_data.items():
                    st.session_state[key] = value
                st.rerun()
            
            # Make prediction if button clicked
            if predict_btn:
                with st.spinner("Analyzing transaction..."):
                    prediction, confidence, details = predict_fraud(input_fields)
                    
                    if prediction is not None:
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Prediction Results")
                        
                        # Create metrics columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            risk_class = "risk-high" if "HIGH" in details['risk_level'] else "risk-medium" if "MEDIUM" in details['risk_level'] else "risk-low"
                            st.markdown(f'<div class="metric-card"><strong>Status:</strong><br><span class="{risk_class}">{prediction}</span></div>', 
                                       unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'<div class="metric-card"><strong>Confidence:</strong><br>{confidence:.4f}</div>', 
                                       unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'<div class="metric-card"><strong>Risk Level:</strong><br>{details["risk_level"]}</div>', 
                                       unsafe_allow_html=True)
                        
                        with col4:
                            fraud_prob = details['fraud_probability'] * 100
                            st.markdown(f'<div class="metric-card"><strong>Fraud Probability:</strong><br>{fraud_prob:.2f}%</div>', 
                                       unsafe_allow_html=True)
                        
                        # Charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(create_probability_chart(details), use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(create_risk_gauge(details), use_container_width=True)
                        
                        # Detailed information
                        with st.expander("📋 Detailed Analysis"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Normal Probability", f"{details['normal_probability']:.4f}")
                                st.metric("Raw Prediction", details['raw_prediction'])
                            with col2:
                                st.metric("Features Used", len(details['features_used']))
                                st.write("**Features:**", ", ".join(details['features_used'][:5]), "...")
    
    elif page == "📊 Model Information":
        st.markdown("---")
        st.markdown("### 🤖 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Information:**
            - **Algorithm:** Random Forest Classifier
            - **Training Dataset:** Credit Card Fraud Detection (Kaggle)
            - **Accuracy:** 99.98%
            - **ROC-AUC:** 99.99%
            - **Features Used:** Top 15 most important features
            - **Class Handling:** SMOTE for imbalance
            """)
        
        with col2:
            st.markdown("""
            **How to Use:**
            1. Enter transaction features (V1-V28 are PCA-transformed features)
            2. Provide Time (in seconds) and Amount
            3. Click "Detect Fraud" to get prediction
            4. Review results and risk assessment
            
            **Important Notes:**
            - V1-V28 are anonymized features from PCA
            - Higher values in certain features may indicate fraud
            - The model considers multiple factors for prediction
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Feature Importance")
        st.info("The model considers these features most important: V14, V10, V17, V4, V12, V3, V16, V2, V9, V7, V11, V21, V19, V8, V27")
        
        # Feature importance chart if available
        if predictor:
            try:
                importance_df = predictor.get_feature_importance()
                if not importance_df.empty:
                    fig = px.bar(
                        importance_df.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Feature Importance",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Feature importance chart not available")
    
    elif page == "🎯 Train Model":
        st.markdown("---")
        st.markdown("### 🎯 Train Custom Fraud Detection Model")
        
        st.markdown("""
        **Upload your credit card transaction dataset to train a custom model:**
        
        **Required CSV format:**
        - `Time`: Time in seconds from first transaction
        - `Amount`: Transaction amount  
        - `V1-V28`: PCA-transformed features (28 columns)
        - `Class`: Target column (0=Normal, 1=Fraud)
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload your dataset (CSV format)", 
            type=['csv'],
            help="Upload a CSV file with credit card transaction data"
        )
        
        if uploaded_file is not None:
            st.info("📊 File uploaded successfully!")
            
            # Show data preview
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head())
                
                # Show data info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Features", len(df.columns) - 1)
                with col3:
                    fraud_count = df['Class'].sum()
                    st.metric("Fraud Cases", fraud_count)
                with col4:
                    fraud_rate = (fraud_count / len(df)) * 100
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
                # Training button
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    with st.spinner("🔄 Training model... This may take a few minutes..."):
                        try:
                            # Reset file pointer
                            uploaded_file.seek(0)
                            
                            # Train model
                            trainer, df_clean = train_from_uploaded_file(uploaded_file)
                            
                            # Show training results
                            st.success("✅ Model trained successfully!")
                            
                            # Display training summary
                            summary = trainer.get_training_summary()
                            
                            st.markdown("#### 📊 Training Results")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Test Accuracy", f"{summary['test_accuracy']:.4f}")
                            with col2:
                                st.metric("ROC AUC Score", f"{summary['roc_auc_score']:.4f}")
                            with col3:
                                st.metric("Features Used", summary['features_used'])
                            with col4:
                                st.metric("Training Samples", summary['training_samples'])
                            
                            # Feature importance chart
                            if trainer.feature_importance is not None:
                                st.markdown("#### 🏆 Feature Importance")
                                fig = px.bar(
                                    trainer.feature_importance.head(15),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Top 15 Most Important Features",
                                    color='Importance',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Success message with next steps
                            st.info("🎉 Your model is ready! Go to 'Fraud Detection' page to start making predictions.")
                            
                            # Add a refresh button for the model
                            if st.button("🔄 Reload Model", help="Click to use the newly trained model"):
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
                            st.error("Please check your data format and try again.")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        else:
            st.info("👆 Upload a CSV file to start training your custom fraud detection model.")
            
            # Show sample format
            st.markdown("#### 📝 Sample Data Format")
            sample_data = {
                'Time': [0, 1, 2],
                'V1': [-1.36, -1.19, -1.36],
                'V2': [-0.07, 0.26, -0.07],
                'V3': [2.54, 0.17, 2.54],
                'V4': [1.38, 0.48, 1.38],
                'V5': [-0.34, -0.15, -0.34],
                'V6': [0.46, 0.09, 0.46],
                'V7': [0.24, 0.06, 0.24],
                'V8': [0.10, -0.06, 0.10],
                'V9': [0.36, 0.14, 0.36],
                'V10': [0.09, -0.21, 0.09],
                'V11': [-0.55, 0.03, -0.55],
                'V12': [-0.62, -0.08, -0.62],
                'V13': [-0.99, 0.10, -0.99],
                'V14': [-0.31, 0.14, -0.31],
                'V15': [1.47, -0.21, 1.47],
                'V16': [-0.47, 0.03, -0.47],
                'V17': [0.21, -0.11, 0.21],
                'V18': [0.03, 0.03, 0.03],
                'V19': [0.40, -0.03, 0.40],
                'V20': [0.25, 0.06, 0.25],
                'V21': [-0.02, -0.06, -0.02],
                'V22': [0.28, 0.01, 0.28],
                'V23': [-0.11, 0.00, -0.11],
                'V24': [0.07, 0.02, 0.07],
                'V25': [0.13, -0.03, 0.13],
                'V26': [-0.19, 0.03, -0.19],
                'V27': [0.13, -0.02, 0.13],
                'V28': [-0.02, 0.03, -0.02],
                'Amount': [149.62, 2.69, 149.62],
                'Class': [0, 0, 0]
            }
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)
    
    elif page == "📈 Feature Importance":
        st.markdown("---")
        st.markdown("### 📈 Understanding the Features")
        
        st.markdown("""
        **Feature Descriptions:**
        
        - **Time:** Number of seconds elapsed between this transaction and the first transaction in the dataset
        - **Amount:** Transaction amount
        - **V1-V28:** Anonymized features resulting from PCA transformation (Principal Component Analysis)
        
        **Key Insights:**
        - V14, V10, and V17 are the most predictive features
        - These features capture patterns in transaction behavior
        - The combination of multiple features provides the best fraud detection
        """)
        
        if predictor:
            try:
                importance_df = predictor.get_feature_importance()
                if not importance_df.empty:
                    st.markdown("### 🏆 Complete Feature Ranking")
                    
                    # Display full feature importance table
                    st.dataframe(
                        importance_df.style.background_gradient(cmap='viridis'),
                        use_container_width=True
                    )
                    
                    # Create detailed importance chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Complete Feature Importance Ranking",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Feature importance data not available")

if __name__ == "__main__":
    main()
