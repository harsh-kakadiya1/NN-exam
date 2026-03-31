"""
Hugging Face Spaces App for Credit Card Fraud Detection
Web interface for users to input transaction data and get fraud predictions
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from predict import FraudDetectionPredictor
import warnings
warnings.filterwarnings('ignore')

# Initialize the predictor
try:
    predictor = FraudDetectionPredictor()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def create_input_fields():
    """Create input fields for all transaction features"""
    
    with gr.Row():
        time_input = gr.Number(label="Time (seconds)", value=0, step=1)
        amount_input = gr.Number(label="Amount ($)", value=100, step=0.01)
    
    with gr.Row():
        v1_input = gr.Number(label="V1", value=0.0, step=0.0001)
        v2_input = gr.Number(label="V2", value=0.0, step=0.0001)
        v3_input = gr.Number(label="V3", value=0.0, step=0.0001)
        v4_input = gr.Number(label="V4", value=0.0, step=0.0001)
    
    with gr.Row():
        v5_input = gr.Number(label="V5", value=0.0, step=0.0001)
        v6_input = gr.Number(label="V6", value=0.0, step=0.0001)
        v7_input = gr.Number(label="V7", value=0.0, step=0.0001)
        v8_input = gr.Number(label="V8", value=0.0, step=0.0001)
    
    with gr.Row():
        v9_input = gr.Number(label="V9", value=0.0, step=0.0001)
        v10_input = gr.Number(label="V10", value=0.0, step=0.0001)
        v11_input = gr.Number(label="V11", value=0.0, step=0.0001)
        v12_input = gr.Number(label="V12", value=0.0, step=0.0001)
    
    with gr.Row():
        v13_input = gr.Number(label="V13", value=0.0, step=0.0001)
        v14_input = gr.Number(label="V14", value=0.0, step=0.0001)
        v15_input = gr.Number(label="V15", value=0.0, step=0.0001)
        v16_input = gr.Number(label="V16", value=0.0, step=0.0001)
    
    with gr.Row():
        v17_input = gr.Number(label="V17", value=0.0, step=0.0001)
        v18_input = gr.Number(label="V18", value=0.0, step=0.0001)
        v19_input = gr.Number(label="V19", value=0.0, step=0.0001)
        v20_input = gr.Number(label="V20", value=0.0, step=0.0001)
    
    with gr.Row():
        v21_input = gr.Number(label="V21", value=0.0, step=0.0001)
        v22_input = gr.Number(label="V22", value=0.0, step=0.0001)
        v23_input = gr.Number(label="V23", value=0.0, step=0.0001)
        v24_input = gr.Number(label="V24", value=0.0, step=0.0001)
    
    with gr.Row():
        v25_input = gr.Number(label="V25", value=0.0, step=0.0001)
        v26_input = gr.Number(label="V26", value=0.0, step=0.0001)
        v27_input = gr.Number(label="V27", value=0.0, step=0.0001)
        v28_input = gr.Number(label="V28", value=0.0, step=0.0001)
    
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

def predict_fraud(time_input, amount_input, v1_input, v2_input, v3_input, v4_input,
                 v5_input, v6_input, v7_input, v8_input, v9_input, v10_input, v11_input, v12_input,
                 v13_input, v14_input, v15_input, v16_input, v17_input, v18_input, v19_input, v20_input,
                 v21_input, v22_input, v23_input, v24_input, v25_input, v26_input, v27_input, v28_input):
    """Make fraud prediction based on user input"""
    
    if not model_loaded:
        return "❌ Model not loaded. Please check model files.", "", "", ""
    
    try:
        # Create transaction dictionary
        transaction_data = {
            'Time': time_input,
            'Amount': amount_input,
            'V1': v1_input, 'V2': v2_input, 'V3': v3_input, 'V4': v4_input,
            'V5': v5_input, 'V6': v6_input, 'V7': v7_input, 'V8': v8_input,
            'V9': v9_input, 'V10': v10_input, 'V11': v11_input, 'V12': v12_input,
            'V13': v13_input, 'V14': v14_input, 'V15': v15_input, 'V16': v16_input,
            'V17': v17_input, 'V18': v18_input, 'V19': v19_input, 'V20': v20_input,
            'V21': v21_input, 'V22': v22_input, 'V23': v23_input, 'V24': v24_input,
            'V25': v25_input, 'V26': v26_input, 'V27': v27_input, 'V28': v28_input
        }
        
        # Make prediction
        prediction, confidence, details = predictor.predict(transaction_data)
        
        # Format results
        result_text = f"## 🎯 Prediction Result\n\n"
        result_text += f"**Transaction Status:** {prediction}\n\n"
        result_text += f"**Confidence:** {confidence:.4f}\n\n"
        result_text += f"**Risk Level:** {details['risk_level']}\n\n"
        result_text += f"**Fraud Probability:** {details['fraud_probability']:.4f}\n\n"
        result_text += f"**Normal Probability:** {details['normal_probability']:.4f}"
        
        # Create probability chart
        fig_prob = go.Figure(data=[
            go.Bar(name='Normal', x=['Probability'], y=[details['normal_probability']], 
                   marker_color='green', text=[f"{details['normal_probability']:.3f}"], textposition='auto'),
            go.Bar(name='Fraud', x=['Probability'], y=[details['fraud_probability']], 
                   marker_color='red', text=[f"{details['fraud_probability']:.3f}"], textposition='auto')
        ])
        fig_prob.update_layout(
            title="Transaction Probability Distribution",
            yaxis_title="Probability",
            barmode='group',
            height=400
        )
        
        # Create gauge chart for fraud risk
        fig_gauge = go.Figure(go.Indicator(
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
        fig_gauge.update_layout(height=400)
        
        return result_text, fig_prob, fig_gauge, "✅ Prediction completed successfully!"
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        return error_msg, None, None, ""

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

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Credit Card Fraud Detection", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🛡️ Credit Card Fraud Detection System
        
        Upload transaction data or input features manually to detect fraudulent transactions.
        This system uses a trained Random Forest model with 99%+ accuracy.
        """)
        
        with gr.Tab("📊 Single Transaction Prediction"):
            gr.Markdown("### Enter Transaction Details")
            
            input_fields = create_input_fields()
            
            with gr.Row():
                predict_btn = gr.Button("🔍 Detect Fraud", variant="primary", size="lg")
                sample_btn = gr.Button("📋 Load Sample Data", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    result_output = gr.Markdown(label="Prediction Results")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    prob_chart = gr.Plot(label="Probability Distribution")
                
                with gr.Column():
                    gauge_chart = gr.Plot(label="Risk Gauge")
        
        with gr.Tab("📈 Model Information"):
            gr.Markdown("""
            ### Model Details
            
            - **Algorithm:** Random Forest Classifier
            - **Training Dataset:** Credit Card Fraud Detection (Kaggle)
            - **Accuracy:** 99%+
            - **Features Used:** Top 15 most important features
            - **Class Handling:** SMOTE for imbalance
            
            ### How to Use
            
            1. Enter transaction features (V1-V28 are PCA-transformed features)
            2. Provide Time (in seconds) and Amount
            3. Click "Detect Fraud" to get prediction
            4. Review results and risk assessment
            
            ### Feature Importance
            
            The model considers these features most important:
            - V14, V4, V12, V10, V11 (top 5)
            - Amount and Time are also significant
            """)
            
            if model_loaded:
                try:
                    importance_df = predictor.get_feature_importance()
                    if not importance_df.empty:
                        fig_importance = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Feature Importance"
                        )
                        gr.Plot(fig_importance)
                except:
                    gr.Markdown("Feature importance chart not available")
        
        # Event handlers
        predict_btn.click(
            fn=predict_fraud,
            inputs=list(input_fields.values()),
            outputs=[result_output, prob_chart, gauge_chart, status_output]
        )
        
        sample_data = load_sample_data()
        sample_btn.click(
            fn=lambda: tuple(sample_data.values()),
            outputs=list(input_fields.values())
        )
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
