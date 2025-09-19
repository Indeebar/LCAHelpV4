# AI-Powered Life Cycle Assessment (LCA) Tool for Metals

An advanced machine learning tool for analyzing and predicting environmental impacts of aluminium and copper production scenarios using Life Cycle Assessment methodologies.

## 🎯 Overview

This project implements a comprehensive LCA analysis tool that:
- Generates realistic synthetic datasets for metal production scenarios
- Trains machine learning models to predict environmental impacts
- Provides explainable AI insights using SHAP analysis
- Supports both aluminium and copper production route analysis

## 📋 Features

### 🔄 Data Generation
- **2000+ synthetic samples** with realistic production scenarios
- **Multiple metal types**: Aluminium and copper
- **Production routes**: Raw material vs recycled material processing
- **Comprehensive features**: Mass, electricity consumption, transport, alloy grades, etc.

### 🎯 Target Predictions
- **GWP (Global Warming Potential)**: CO₂ equivalent emissions
- **Energy Consumption**: Total energy in MJ
- **Circularity Index**: Sustainability metric (0-100 scale)

### 🤖 Machine Learning Models
- **Random Forest Regressor**: Ensemble method for robust predictions
- **XGBoost Regressor**: Gradient boosting for high performance
- **Multi-output regression**: Simultaneous prediction of all targets
- **Comprehensive evaluation**: MAE, RMSE, R² metrics

### 📊 Explainability
- **SHAP analysis**: Feature importance and contribution analysis
- **Visual insights**: Top feature plots and summary visualizations
- **Model interpretability**: Understand which factors drive environmental impact

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

### Alternative Installation
```bash
# Using conda
conda install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install shap
```

## 🚀 Quick Start

### Option A: Using Jupyter Interface

#### 1. Clone or Download
Download the `LCA_Analysis_Notebook.ipynb` file to your working directory.

#### 2. Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

#### 3. Open the Notebook
Navigate to and open `LCA_Analysis_Notebook.ipynb`

#### 4. Run All Cells
Execute all cells sequentially from top to bottom.

### Option D: Google Colab (Cloud Execution)

#### 1. Upload Notebook to Colab
```bash
# Go to https://colab.research.google.com/
# Click "Upload" and select LCA_Analysis_Notebook.ipynb
# OR use this direct link format:
# https://colab.research.google.com/github/your-repo/LCA_Analysis_Notebook.ipynb
```

#### 2. Install Required Packages (First Cell)
```python
# Add this as the first cell in Colab
!pip install xgboost shap
# Note: pandas, numpy, scikit-learn, matplotlib are pre-installed in Colab
```

#### 3. Run All Cells
```bash
# In Colab: Runtime → Run All
# Or use Ctrl+F9 (Windows) / Cmd+F9 (Mac)
```

#### 4. Download Results
```python
# Add this cell at the end to download files
from google.colab import files

# Download generated files
files.download('synthetic_LCA.csv')
files.download('lca_rf_model.pkl')
files.download('lca_xgb_model.pkl')
files.download('processed_features.csv')
files.download('processed_targets.csv')
```

#### 5. Mount Google Drive (Optional)
```python
# To save files directly to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Modify save paths in the notebook
# Example: df.to_csv('/content/drive/MyDrive/synthetic_LCA.csv')
```

### Option C: Direct Terminal Execution (Qoder IDE)

#### 1. Navigate to Project Directory
```bash
# In Qoder terminal
cd f:\LCAHelp
```

#### 2. Install Required Packages
```bash
# Install all dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter nbconvert
```

#### 3. Execute Notebook Directly in Terminal
```bash
# Convert notebook to Python script and execute
jupyter nbconvert --to script LCA_Analysis_Notebook.ipynb
python LCA_Analysis_Notebook.py

# OR execute notebook directly (recommended)
jupyter nbconvert --to notebook --execute LCA_Analysis_Notebook.ipynb --output LCA_Results.ipynb

# OR execute and generate HTML report
jupyter nbconvert --to html --execute LCA_Analysis_Notebook.ipynb
```

#### 4. Alternative: Using papermill (Advanced)
```bash
# Install papermill for notebook execution
pip install papermill

# Execute notebook with papermill
papermill LCA_Analysis_Notebook.ipynb output_notebook.ipynb
```

#### 5. Check Results
```bash
# List all generated files
dir *.csv *.pkl *.html *.ipynb
```

### Option B: Terminal Commands (Step-by-Step)

#### 1. Navigate to Project Directory
```bash
# Windows (PowerShell/Command Prompt)
cd f:\LCAHelp

# Linux/macOS
cd /path/to/LCAHelp
```

#### 2. Install Required Packages
```bash
# Install all required packages
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter

# Or using conda (if you prefer)
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
pip install xgboost shap
```

#### 3. Launch Jupyter Notebook
```bash
# Start Jupyter Notebook server
jupyter notebook

# Alternative: Use JupyterLab (more modern interface)
jupyter lab

# If you want to specify a port
jupyter notebook --port=8888
```

#### 4. Access the Notebook
- Your browser will automatically open to `http://localhost:8888`
- Click on `LCA_Analysis_Notebook.ipynb` to open it
- Run all cells sequentially (Cell → Run All)

#### 5. Alternative: Run Notebook from Terminal
```bash
# Convert and execute notebook non-interactively
jupyter nbconvert --to notebook --execute LCA_Analysis_Notebook.ipynb

# Or execute and save output
jupyter nbconvert --to html --execute LCA_Analysis_Notebook.ipynb
```

#### 6. Verify Output Files
```bash
# List generated files
ls -la *.csv *.pkl

# Windows equivalent
dir *.csv *.pkl
```

The notebook will:
- Generate synthetic data
- Train ML models
- Perform analysis
- Save results

## 📁 File Structure

After running the notebook, you'll have:

```
LCAHelp/
├── LCA_Analysis_Notebook.ipynb    # Main analysis notebook
├── README.md                      # This file
├── synthetic_LCA.csv             # Generated synthetic dataset
├── lca_rf_model.pkl              # Trained Random Forest model
├── lca_xgb_model.pkl             # Trained XGBoost model
├── preprocessing_info.pkl         # Preprocessing objects
├── processed_features.csv         # Processed feature matrix
└── processed_targets.csv          # Target variables
```

## 📊 Dataset Features

### Input Features
| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `metal` | Categorical | Metal type | aluminium, copper |
| `route` | Categorical | Production route | raw, recycled |
| `mass_kg` | Numeric | Product mass | 500-5000 kg |
| `electricity_kWh` | Numeric | Electricity consumption | 200-2000 kWh |
| `grid_co2_g_per_kWh` | Numeric | Grid carbon intensity | 100-1000 g CO₂/kWh |
| `transport_mode` | Categorical | Transportation method | truck, rail, ship |
| `transport_km` | Numeric | Transport distance | 10-2000 km |
| `yield_frac` | Numeric | Production yield | 0.5-1.0 |
| `recycled_input_frac` | Numeric | Recycled content | 0-1 |
| `end_of_life_recovery_frac` | Numeric | End-of-life recovery | 0-1 |
| `alloy_grade` | Categorical | Alloy specification | 6061, 1100, Cu-ETP, etc. |

### Target Variables
| Target | Description | Units |
|--------|-------------|--------|
| `GWP_kgCO2e` | Global Warming Potential | kg CO₂ equivalent |
| `energy_MJ` | Total Energy Consumption | MJ |
| `circularity_index` | Sustainability Score | 0-100 scale |

## 🎯 Model Performance

The notebook trains and compares two models:

### Random Forest
- **Advantages**: Robust, handles non-linear relationships well
- **Good for**: Feature importance analysis, stable predictions

### XGBoost
- **Advantages**: High performance, efficient training
- **Good for**: Complex pattern recognition, competitive accuracy

Performance metrics include:
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **R² Score**: Proportion of variance explained

## 🔍 Explainability Analysis

### SHAP (SHapley Additive exPlanations)
- **Feature importance**: Which factors most influence each target
- **Feature contributions**: How each feature affects individual predictions
- **Visual analysis**: Bar plots and summary plots for interpretation

### Key Insights
The analysis typically reveals:
- **Electricity consumption** is the primary driver of GWP and energy
- **Recycled content** significantly reduces environmental impact
- **Transport distance** affects both emissions and energy consumption
- **Metal type** influences baseline environmental performance

## 💡 Usage Examples

### Loading Trained Models
```python
import pickle
import pandas as pd

# Load trained model
with open('lca_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessing info
with open('preprocessing_info.pkl', 'rb') as f:
    preprocessing_info = pickle.load(f)

# Make predictions on new data
new_data = pd.DataFrame({...})  # Your new data
predictions = model.predict(new_data)
```

### Analyzing Feature Importance
```python
import shap

# Load model and data
# ... (loading code)

# Create SHAP explainer
explainer = shap.TreeExplainer(model.estimators_[0])  # For first target
shap_values = explainer.shap_values(X_sample)

# Plot feature importance
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
```

## 🔧 Customization

### Modifying Dataset Size
```python
# Change the number of samples
df_synthetic = generate_synthetic_lca_data(n_samples=5000)
```

### Adding New Features
Modify the `generate_synthetic_lca_data()` function to include additional features relevant to your LCA analysis.

### Tuning Model Parameters
```python
# Customize Random Forest
rf_model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200,  # More trees
    max_depth=20,      # Deeper trees
    random_state=42
))
```

## 📈 Interpretation Guide

### GWP (Global Warming Potential)
- **Lower is better**: Indicates reduced climate impact
- **Key drivers**: Electricity consumption, grid carbon intensity
- **Reduction strategies**: Use renewable energy, increase recycled content

### Energy Consumption
- **Lower is better**: Indicates improved energy efficiency
- **Key drivers**: Electricity use, transport distance
- **Reduction strategies**: Optimize processes, reduce transport

### Circularity Index
- **Higher is better**: Indicates better sustainability
- **Key drivers**: Recycled input, end-of-life recovery
- **Improvement strategies**: Increase recycling, design for circularity

## 🚨 Limitations

- **Synthetic data**: Generated data may not capture all real-world complexities
- **Simplified formulas**: Actual LCA calculations are more complex
- **Regional variations**: Grid carbon intensity varies by location
- **Technology changes**: Real production technologies evolve over time

## 🤝 Contributing

To extend this tool:
1. **Add real data**: Replace synthetic data with actual LCA databases
2. **Include more metals**: Extend to steel, zinc, etc.
3. **Add impact categories**: Include water use, land use, etc.
4. **Improve models**: Experiment with neural networks, ensemble methods

## 📚 References

- Life Cycle Assessment methodology (ISO 14040/14044)
- Machine Learning for Environmental Science
- SHAP documentation: https://shap.readthedocs.io/
- Scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Review the notebook comments and documentation
2. Check that all required packages are installed
3. Ensure you're running cells in sequence
4. Verify Python version compatibility (3.7+)

---

**Note**: This tool is designed for educational and research purposes. For production LCA studies, consider using established LCA databases and software tools in conjunction with these machine learning insights.