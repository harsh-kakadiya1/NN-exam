# 🚀 Deployment Guide for Credit Card Fraud Detection App

## 📋 Prerequisites

Before deploying your Streamlit app, ensure you have:

1. **All model files** in the same directory:
   - `fraud_detection_model.pkl`
   - `feature_scaler.pkl`
   - `selected_features.pkl`
   - `streamlit_app.py`
   - `predict.py`
   - `requirements.txt`

2. **Python environment** with required packages installed

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Model Files
Make sure all pickle files are present:
```bash
ls -la *.pkl
# Should show:
# - fraud_detection_model.pkl
# - feature_scaler.pkl
# - selected_features.pkl
```

## 🌐 Deployment Options

### Option 1: Local Development
Run the app locally:
```bash
streamlit run streamlit_app.py
```
Access at: `http://localhost:8501`

### Option 2: Streamlit Community Cloud
1. **Create a GitHub repository** with all files
2. **Go to [Streamlit Community Cloud](https://share.streamlit.io/)**
3. **Connect your GitHub account**
4. **Create new app** from your repository
5. **Select `streamlit_app.py` as main file**

### Option 3: Heroku Deployment
1. **Install Heroku CLI**
2. **Create `Procfile`**:
   ```
   web: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
   ```
3. **Create `runtime.txt`**:
   ```
   python-3.9.16
   ```
4. **Deploy**:
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Deploy fraud detection app"
   git push heroku main
   ```

### Option 4: Railway
1. **Install Railway CLI**
2. **Login to Railway**
3. **Deploy**:
   ```bash
   railway login
   railway init
   railway up
   ```

### Option 5: PythonAnywhere
1. **Create a PythonAnywhere account**
2. **Create a new Web app**
3. **Upload your files**
4. **Install requirements in virtual environment**
5. **Set up WSGI file**:
   ```python
   import sys
   import os
   import subprocess
   
   # Add your project directory to Python path
   path = '/home/yourusername/your-project-folder'
   if path not in sys.path:
       sys.path.append(path)
   
   # Start Streamlit
   subprocess.run(['streamlit', 'run', 'streamlit_app.py', '--server.port', '8080'])
   ```

## 🔧 Configuration Options

### Environment Variables
You can set these environment variables:

- `STREAMLIT_SERVER_PORT`: Change default port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Change server address (default: localhost)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (true/false)

### Custom Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## 🚨 Troubleshooting

### Common Issues & Solutions

1. **Model Loading Error**
   ```
   Solution: Ensure all .pkl files are in the same directory
   ```

2. **Port Already in Use**
   ```bash
   # Kill existing Streamlit process
   pkill -f streamlit
   
   # Or use different port
   streamlit run streamlit_app.py --server.port 8502
   ```

3. **Import Errors**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **Memory Issues**
   ```bash
   # Limit memory usage
   streamlit run streamlit_app.py --server.maxMessageSize 200
   ```

5. **CORS Issues**
   ```bash
   # Enable CORS
   streamlit run streamlit_app.py --server.enableCORS false
   ```

## 📊 Monitoring & Analytics

### Add Google Analytics (Optional)
Add to `.streamlit/config.toml`:
```toml
[browser]
gatherUsageStats = true
```

### Log User Actions
Add to your `streamlit_app.py`:
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
def log_prediction(input_data, prediction):
    logger.info(f"Prediction made: {prediction} for input: {input_data}")
```

## 🔒 Security Considerations

1. **Input Validation**: The app includes basic input validation
2. **Rate Limiting**: Consider implementing rate limiting for production
3. **HTTPS**: Always use HTTPS in production
4. **Authentication**: Add user authentication for sensitive applications

## 📈 Performance Optimization

1. **Caching**: The app uses `@st.cache_resource` for model loading
2. **Lazy Loading**: Components are loaded only when needed
3. **Optimized Charts**: Using Plotly for interactive but performant visualizations

## 🌍 Multi-language Support

To add multiple languages, create a language dictionary:
```python
translations = {
    'en': {'title': 'Fraud Detection', 'predict': 'Predict'},
    'es': {'title': 'Detección de Fraude', 'predict': 'Predecir'},
    # Add more languages
}
```

## 📱 Mobile Responsiveness

The app is designed to be mobile-responsive with:
- Responsive columns
- Touch-friendly buttons
- Adaptive charts

## 🔄 Continuous Deployment

### GitHub Actions (for automated deployment)
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit
      uses: streamlit/streamlit-app-action@v0.0.1
      with:
        app-name: your-app-name
        repo: your-username/your-repo
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all files are present and correctly formatted
3. Check Streamlit logs: `streamlit logs`
4. Visit [Streamlit documentation](https://docs.streamlit.io/)

## 🎉 Success!

Your fraud detection app is now ready for production! Users can:
- Input transaction data
- Get real-time fraud predictions
- View detailed analysis and risk assessment
- Understand feature importance

The app provides a professional, user-friendly interface for your ML model with comprehensive visualizations and detailed results.
