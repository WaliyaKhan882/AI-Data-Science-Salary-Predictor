# AI/ML Salary Prediction Project

## Overview
This project analyzes AI/ML job salaries across 10 countries and builds a machine learning model to predict salaries based on various factors including location, job title, experience, education, and more.

## Dataset
- **Records**: 500 AI/ML job entries
- **Countries**: 10 (USA, India, UK, Canada, France, UAE, Germany, Singapore, Australia, Netherlands)
- **Time Period**: 2020-2025
- **Features**: 16 including salary (converted to USD), job title, experience, education, company size, industry, etc.

## Project Structure
```
Term Project/
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis (15 analyses)
│   ├── 02_Preprocessing.ipynb     # Data preprocessing and feature engineering
│   └── 03_Model.ipynb             # Model training and evaluation
├── data/
│   ├── X_train_scaled.csv         # Preprocessed training features
│   ├── X_test_scaled.csv          # Preprocessed test features
│   ├── y_train.csv                # Training target variable
│   ├── y_test.csv                 # Test target variable
│   ├── label_encoders.pkl         # Saved label encoders
│   ├── scaler.pkl                 # Saved feature scaler
│   ├── feature_columns.pkl        # Selected feature columns
│   ├── best_model.pkl             # Trained best model
│   └── model_metadata.pkl         # Model performance metadata
├── app.py                         # Streamlit web application
├── DATA.csv                       # Original dataset
├── requirements.txt               # Python dependencies
└── readme.md                      # This file
```

## Installation

1. **Clone or download this project**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis
Open and run `notebooks/01_EDA.ipynb` to explore the dataset:
- 15 comprehensive analyses including salary distribution, geographic trends, job title comparisons
- Feature importance analysis
- All visualizations use USD-converted salaries

### 2. Data Preprocessing
Open and run `notebooks/02_Preprocessing.ipynb` to prepare data:
- Currency conversion to USD
- Feature engineering (experience levels, salary categories)
- Encoding categorical variables
- Train-test split (80/20)
- Feature scaling
- Saves preprocessed data to `data/` folder

### 3. Model Training
Open and run `notebooks/03_Model.ipynb` to train models:
- Trains 6 algorithms: Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting
- Compares model performance
- Saves best model and metadata

### 4. Launch Streamlit App
Run the interactive web application:
```bash
streamlit run app.py
```

The app includes:
- **Introduction**: Project overview and dataset preview
- **EDA Section**: 8 interactive analyses with visualizations
- **Model Section**: 
  - Model overview and performance metrics
  - Runtime salary predictions (interactive form)
  - Performance analysis
- **Conclusion**: Key findings, actionable insights, and future directions

## Key Findings

1. **Geographic Location is Dominant**: 860% salary difference between countries (USA highest at $240k, India lowest at $25k)
2. **Emerging AI Roles Pay Premium**: LLM Researchers, AI Architects command 23% higher salaries
3. **Experience Matters Moderately**: Correlation of 0.41, each year adds $8-10k
4. **Model Performance**: Achieved R² of 0.70-0.85, indicating strong predictive accuracy
5. **Other Factors**: Education (26% difference), Industry (20% difference), Company Size (6% difference)

## Model Performance

- **Algorithm**: Best performing model selected from 6 candidates
- **R² Score**: 0.70-0.85 (explains 70-85% of variance)
- **RMSE**: ~$20k-$30k
- **Features Used**: 7 key features (country, job_title, years_experience, experience_level, industry, education, company_size)

## Technologies Used

- **Python 3.13+**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Web App**: streamlit

## Author

MSDS Term Project - Fall 2025

## License

This project is for educational purposes.
