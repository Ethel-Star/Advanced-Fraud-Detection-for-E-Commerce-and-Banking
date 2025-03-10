## Data Analysis and Preprocessing

### Handling Missing Values

- Impute or drop missing values
- Remove duplicate entries
- Correct data types

### Exploratory Data Analysis (EDA)

- Univariate analysis
- Bivariate analysis

### Merging Datasets for Geolocation Analysis

- Convert IP addresses to integer format
- Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`

### Feature Engineering

- Transaction frequency and velocity
- Time-based features:
  - `hour_of_day`
  - `day_of_week`

### Normalization and Scaling

- Encode categorical features

## Model Building and Training

### Data Preparation

- Separate features and target variables (`Class` in `creditcard.csv`, `class` in `Fraud_Data.csv`)
- Train-test split

### Model Selection

Multiple models are tested for performance:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

### Model Training and Evaluation

- Train models on both datasets

### MLOps Steps

- Use MLflow for experiment tracking, logging parameters, metrics, and model versioning.

## Model Explainability

### SHAP (Shapley Additive exPlanations)

- Summary Plot
- Force Plot
- Dependence Plot

### LIME (Local Interpretable Model-agnostic Explanations)

- Feature Importance Plot

## Model Deployment and API Development

### Flask API Setup

- `serve_model.py` for serving predictions
- `requirements.txt` listing dependencies

### API Development

- Define API endpoints
- Test API functionality