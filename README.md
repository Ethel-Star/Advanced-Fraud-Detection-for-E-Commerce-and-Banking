# Advanced-Fraud-Detection-for-E-Commerce-and-Banking
## Project Overview

Adey Innovations Inc. is committed to enhancing fraud detection mechanisms for e-commerce transactions and bank credit transactions. This project aims to develop robust machine learning models for fraud detection, leveraging geolocation analysis and transaction pattern recognition. The solution includes advanced model interpretability, real-time monitoring, and API-based deployment for integration into financial systems.

## Table of Contents

1. [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
2. [Model Building and Training](#model-building-and-training)
3. [Model Explainability](#model-explainability)
4. [Model Deployment and API Development](#model-deployment-and-api-development)
5. [Dashboard Development](#dashboard-development)
6. [Business Impact](#business-impact)
7. [Installation and Usage](#installation-and-usage)

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

### Dockerization

**Dockerfile:**

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "serve_model.py"]
```

**Commands:**

```sh
docker build -t fraud-detection-model .
docker run -p 5000:5000 fraud-detection-model
```

- Integrate logging for tracking API requests and fraud predictions.

## Dashboard Development

Using Flask and Dash:

- Flask backend serves fraud data via API endpoints.
- Dash frontend visualizes insights.

### Dashboard Insights

- Summary boxes for total transactions, fraud cases, and fraud percentage.
- Line chart tracking fraud trends over time.
- Geolocation fraud analysis.
- Bar chart comparing fraud cases by device and browser.

## Business Impact

- Improved fraud detection accuracy for banking and e-commerce.
- Enhanced transaction security, reducing financial losses.
- Real-time fraud monitoring and risk mitigation.
- Strengthened customer trust in financial institutions.

## Installation and Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/Ethel-Star/Advanced-Fraud-Detection-for-E-Commerce-and-Banking.git
   cd Advanced-Fraud-Detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run Flask API:
   ```sh
   python serve_model.py
   ```
4. Deploy with Docker:
   ```sh
   docker build -t fraud-detection-model .
   docker run -p 5000:5000 fraud-detection-model
   ```
5. Start Dashboard:
   ```sh
   python dashboard.py
   ```

## License

This project is licensed under the Apache License


