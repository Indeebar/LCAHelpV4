# 🚀 Web Application Deployment Guide

## Overview
This guide helps you deploy your AI-Driven LCA Sustainability Predictor as a web application using Streamlit.

## Files Created
- `lca_web_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `deploy_instructions.md` - This file

## Prerequisites
1. Trained model files from Google Colab:
   - `best_lca_model.joblib`
   - `feature_scaler.joblib`
2. Python 3.8+ installed

## Local Deployment (Testing)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Copy Model Files
Ensure these files are in the same directory as `lca_web_app.py`:
- `best_lca_model.joblib` (from your Colab notebook)
- `feature_scaler.joblib` (from your Colab notebook)

### Step 3: Run the Application
```bash
streamlit run lca_web_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Upload to GitHub**:
   - Create a new GitHub repository
   - Upload all files: `lca_web_app.py`, `requirements.txt`, model files

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Click "Deploy"

### Option 2: Heroku

1. **Create Heroku App**:
   ```bash
   heroku create your-lca-app-name
   ```

2. **Add Procfile**:
   Create `Procfile` with:
   ```
   web: sh setup.sh && streamlit run lca_web_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Add setup.sh**:
   Create `setup.sh` with:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

4. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy LCA app"
   git push heroku main
   ```

### Option 3: Google Cloud Run

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8080
   CMD streamlit run lca_web_app.py --server.port=8080 --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   gcloud run deploy lca-predictor --source . --platform managed --region us-central1 --allow-unauthenticated
   ```

## Application Features

### 🎯 Interactive Interface
- **Sidebar Controls**: Easy parameter input
- **Real-time Predictions**: Instant sustainability scoring
- **Visual Feedback**: Color-coded results
- **Recommendations**: Actionable improvement suggestions

### 📊 Key Functionalities
- **Metal Type Selection**: Aluminum or Copper
- **Process Route**: Raw vs Recycled materials
- **Environmental Parameters**: Energy, transport, emissions
- **Circularity Metrics**: Recycled content, recovery rates
- **Sustainability Scoring**: 0-100 scale with categories

### 🔍 Analysis Features
- **Radar Chart**: Multi-dimensional process analysis
- **Impact Metrics**: GWP, Energy, Circularity potential
- **Optimization Tips**: Personalized recommendations
- **Process Comparison**: Different scenarios

## Troubleshooting

### Model Files Not Found
```
Error: Model files not found!
```
**Solution**: Ensure `best_lca_model.joblib` and `feature_scaler.joblib` are in the app directory.

### Import Errors
```
ModuleNotFoundError: No module named 'package_name'
```
**Solution**: Install missing packages:
```bash
pip install package_name
```

### Port Issues (Local)
```
Port 8501 is already in use
```
**Solution**: Use different port:
```bash
streamlit run lca_web_app.py --server.port 8502
```

### Memory Issues (Large Models)
For deployment platforms with memory limits, consider:
- Model compression techniques
- Lazy loading of models
- Using lighter model formats

## Usage Instructions for Demo

1. **Basic Demo Flow**:
   - Select metal type (aluminum/copper)
   - Choose production route (raw/recycled)
   - Adjust process parameters
   - Click "Predict Sustainability"
   - Review results and recommendations

2. **Showcase Features**:
   - Compare raw vs recycled routes
   - Show impact of renewable energy (low CO₂ grid)
   - Demonstrate recycled content benefits
   - Highlight optimization recommendations

3. **Demo Scenarios**:
   
   **High Sustainability Scenario**:
   - Metal: Aluminum
   - Route: Recycled
   - High recycled content (80%+)
   - Low grid CO₂ intensity
   - High end-of-life recovery
   
   **Low Sustainability Scenario**:
   - Metal: Copper
   - Route: Raw
   - Low recycled content
   - High grid CO₂ intensity
   - Long transport distances

## Customization Options

### Branding
- Update colors in CSS section
- Change title and descriptions
- Add company logos

### Features
- Add more input parameters
- Include additional predictions (cost, time)
- Integrate with databases
- Add user authentication

### Analytics
- Add usage tracking
- Include performance metrics
- Create admin dashboards

## Support

For technical issues:
1. Check model file locations
2. Verify Python environment
3. Review error logs
4. Test with different input combinations

The web application provides a professional, interactive interface perfect for demonstrating your AI-powered LCA prediction system!