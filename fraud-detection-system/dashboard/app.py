import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import os
import requests
import json
import time
from datetime import datetime, timedelta

# Initialize Dash app
app = dash.Dash(__name__, assets_folder='assets', title='Fraud Detection Dashboard')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'api/data/featured_fraud_data.csv')

# Load fraud data
try:
    df = pd.read_csv(DATA_PATH)
    country_cols = [col for col in df.columns if col.startswith('ctry_')]
    if country_cols:
        df['country'] = df[country_cols].idxmax(axis=1).str.replace('ctry_', '').str.replace('_', ' ')
        df['country'] = df['country'].where(df[country_cols].sum(axis=1) == 1, 'Unknown')
    else:
        print("Warning: No 'ctry_*' columns found in data. Adding simulated geographical data.")
        country_list = ['USA', 'India', 'China', 'Brazil', 'Germany']
        df['country'] = [country_list[i % len(country_list)] for i in range(len(df))]
except FileNotFoundError:
    print("Warning: Data file not found. Using simulated data with geographical information.")
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='H')
    np.random.seed(42)
    n_transactions = 5000
    indices = np.linspace(0, len(dates) - 1, n_transactions).astype(int)
    selected_dates = dates[indices]
    weeks = pd.date_range(start='2023-01-01', end='2023-06-30', freq='W')
    fraud_per_week = np.random.uniform(5, 15, size=len(weeks))
    fraud_indices = []
    for i, week_start in enumerate(weeks):
        week_end = week_start + timedelta(days=6)
        week_indices = np.where((selected_dates >= week_start) & (selected_dates <= week_end))[0]
        n_fraud = int(fraud_per_week[i])
        if len(week_indices) > n_fraud:
            fraud_indices.extend(np.random.choice(week_indices, n_fraud, replace=False))
    df = pd.DataFrame({
        'purchase_timestamp': selected_dates,
        'device_id': np.random.choice(['device_1', 'device_2', 'device_3', 'device_4'], size=n_transactions),
        'class': 0,
        'br_FireFox': np.random.choice([0, 1], size=n_transactions, p=[0.8, 0.2]),
        'br_IE': np.random.choice([0, 1], size=n_transactions, p=[0.9, 0.1]),
        'br_Opera': np.random.choice([0, 1], size=n_transactions, p=[0.95, 0.05]),
        'br_Safari': np.random.choice([0, 1], size=n_transactions, p=[0.7, 0.3]),
        'amount': np.random.uniform(10, 1000, size=n_transactions),
    })
    df.loc[fraud_indices, 'class'] = 1
    df['purchase_hour'] = df['purchase_timestamp'].dt.hour
    country_list = ['USA', 'India', 'China', 'Brazil', 'Germany']
    df['country'] = [country_list[i % len(country_list)] for i in range(n_transactions)]

# Filter fraud cases
fraud_df = df[df['class'] == 1]

# Summary Statistics
total_transactions = len(df)
fraud_count = fraud_df.shape[0]
fraud_percentage = round((fraud_count / total_transactions) * 100, 2)

# Time Series Chart (Fraud Cases by Purchase Hour)
df['purchase_timestamp'] = pd.to_datetime(df['purchase_timestamp'], errors='coerce')
df = df.dropna(subset=['purchase_hour'])
print("Purchase Hour Range:", df['purchase_hour'].min(), "to", df['purchase_hour'].max())
print("Total Fraud Cases:", df['class'].sum())
time_series = df.groupby('purchase_hour')['class'].sum().reset_index()
print("Time Series Data (by Hour):", time_series)
if not time_series.empty and len(time_series) > 1:
    fig_time = px.line(
        time_series,
        x='purchase_hour',
        y='class',
        title='Fraud Cases by Purchase Hour',
        template='plotly',  # Use light theme
        color_discrete_sequence=['#FF6B97'],
        labels={'class': 'Fraud Cases', 'purchase_hour': 'Hour of Day (0-23)'},
    )
    fig_time.update_traces(line=dict(width=3))
    fig_time.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(size=20, color='#FF6B97'),
        xaxis=dict(tickmode='linear', dtick=1),
    )
else:
    print("Warning: Not enough data for time series plot. Falling back to bar plot.")
    fig_time = px.bar(
        time_series,
        x='purchase_hour',
        y='class',
        title='Fraud Cases by Purchase Hour - Fallback',
        template='plotly',
        color_discrete_sequence=['#FF6B97'],
        labels={'class': 'Fraud Cases', 'purchase_hour': 'Hour of Day (0-23)'},
    )
    fig_time.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(size=20, color='#FF6B97'),
        xaxis=dict(tickmode='linear', dtick=1),
    )

# Device Chart (Fraud by Device)
device_counts = fraud_df.groupby('device_id')['class'].count().reset_index()
device_counts = device_counts.rename(columns={'class': 'fraud_count'})
if device_counts.empty:
    device_counts = pd.DataFrame({'device_id': ['No Devices'], 'fraud_count': [0]})
fig_device = px.bar(
    device_counts,
    x='device_id',
    y='fraud_count',
    title='Fraud by Device',
    template='plotly',
    color='device_id',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    labels={'fraud_count': 'Fraud Cases', 'device_id': 'Device ID'},
)
fig_device.update_traces(marker_line_color='#333333', marker_line_width=1.5)
fig_device.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#333333'),
    title_font=dict(size=20, color='#FF6B97'),
    showlegend=True,
)

# Browser Chart (Fraud by Browser)
browser_cols = ['br_FireFox', 'br_IE', 'br_Opera', 'br_Safari']
browser_cols = [col for col in browser_cols if col in fraud_df.columns]
browser_counts = fraud_df[browser_cols].sum()
browser_counts_df = pd.DataFrame({'browser': browser_cols, 'fraud_count': browser_counts})
fig_browser = px.bar(
    browser_counts_df,
    x='browser',
    y='fraud_count',
    title='Fraud by Browser',
    template='plotly',
    color='browser',
    color_discrete_sequence=px.colors.qualitative.Safe,
    labels={'fraud_count': 'Fraud Cases', 'browser': 'Browser'},
)
fig_browser.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#333333'),
    title_font=dict(size=20, color='#FF6B97'),
)

# Geographical Analysis (Fraud by Country)
if 'country' in fraud_df.columns:
    geo_counts = fraud_df.groupby('country').size().reset_index(name='fraud_count')
    fig_geo = px.choropleth(
        geo_counts,
        locations='country',
        locationmode='country names',
        color='fraud_count',
        hover_name='country',
        title='Geographical Distribution of Fraud Cases',
        template='plotly',
        color_continuous_scale='Plasma',
        labels={'fraud_count': 'Fraud Cases'},
    )
    fig_geo.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(size=20, color='#FF6B97'),
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            landcolor='#E6E6FA',
            showcountries=True,
            countrycolor='#FFFFFF',
        ),
    )
else:
    fig_geo = px.scatter()
    fig_geo.update_layout(
        title='Geographical Data Unavailable',
        template='plotly',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(size=20, color='#FF6B97'),
    )

# Fetch API features for JSON input validation
def get_api_features():
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('credit_features', []), data.get('fraud_features', [])
    except requests.exceptions.RequestException:
        return [], []

credit_features, fraud_features = get_api_features()

# Generate default JSON for placeholders
def generate_default_json(features, defaults=None):
    if not features:
        return "{}"
    if defaults is None:
        defaults = {}
    json_dict = {}
    for feature in features:
        json_dict[feature] = defaults.get(feature, 0 if feature.startswith(('src_', 'br_', 'ctry_')) else 1)
    return json.dumps(json_dict, indent=2)

credit_defaults = {
    "Time": 123456,
    "Amount": 250.75,
    "scaled_amount": 0.25,
    "time_hour": 14,
    "day_of_week": 3,
    "is_weekend": 0
}
fraud_defaults = {
    "purchase_value": 750.00,
    "age": 35,
    "time_to_purchase": 3600,
    "purchase_hour": 14,
    "purchase_day": 3,
    "signup_hour": 10,
    "transaction_frequency": 10,
    "transaction_velocity": 3.5,
    "age_scaled": 0.5,
    "src_Direct": 1,
    "br_FireFox": 1,
    "ctry_India": 1,
    "sex_code": 1
}

credit_json_placeholder = generate_default_json(credit_features, credit_defaults)
fraud_json_placeholder = generate_default_json(fraud_features, fraud_defaults)

# Layout with logo and CSS classes
app.layout = html.Div(className='main-container', children=[
    html.Div(className='logo-container', children=[
        html.Img(src='/assets/logo.png', alt='10 Academy: AI Mastery Class Logo')
    ]),
    html.H2("Top 10 Academy: Artificial Intelligence Mastery Class", className='gradient-text'),
    html.H1("Advanced Fraud Detection for E-commerce and Bank Transactions"),
    html.H3("Developed by Ethel.c"),
    html.Div(className='nav-buttons', children=[
        html.Button("Summary Dashboard", id='summary-button', n_clicks=0),
        html.Button("ML Prediction", id='prediction-button', n_clicks=0),
    ]),
    html.Div(id='content-area'),
])

@app.callback(
    Output('content-area', 'children'),
    [Input('summary-button', 'n_clicks'),
     Input('prediction-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_content(summary_clicks, prediction_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div()

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'summary-button':
        return html.Div([
            html.Div(className='summary-section', children=[
                html.H2("Summary Statistics"),
                html.Div(className='metrics-row', children=[
                    html.Div(className='metric-box', children=[
                        html.P("Total Transactions"),
                        html.H3(f"{total_transactions:,}")
                    ]),
                    html.Div(className='metric-box', children=[
                        html.P("Fraud Cases"),
                        html.H3(f"{fraud_count:,}")
                    ]),
                    html.Div(className='metric-box', children=[
                        html.P("Fraud Percentage"),
                        html.H3(f"{fraud_percentage}%")
                    ]),
                ]),
            ]),
            html.Div(className='visualization-section', children=[
                html.H2("Fraud Insights"),
                dcc.Graph(figure=fig_time),
                dcc.Graph(figure=fig_device),
                dcc.Graph(figure=fig_browser),
                dcc.Graph(figure=fig_geo),
            ]),
        ])

    elif button_id == 'prediction-button':
        return html.Div([
            html.Div(className='prediction-section', children=[
                html.H2("Prediction"),
                html.Div([
                    html.H3("Credit Prediction"),
                    html.Label("Required Features: "),
                    html.P(", ".join(credit_features) if credit_features else "Unable to fetch features. Ensure API is running."),
                    html.Label("Enter Features (JSON format):"),
                    dcc.Textarea(
                        id='credit-input',
                        value=credit_json_placeholder,
                    ),
                    html.Button('Predict Credit', id='credit-predict-button', n_clicks=0),
                    html.Div(id='credit-output'),
                ]),
                html.Div([
                    html.H3("Fraud Prediction"),
                    html.Label("Required Features: "),
                    html.P(", ".join(fraud_features) if fraud_features else "Unable to fetch features. Ensure API is running."),
                    html.Label("Enter Features (JSON format):"),
                    dcc.Textarea(
                        id='fraud-input',
                        value=fraud_json_placeholder,
                    ),
                    html.Button('Predict Fraud', id='fraud-predict-button', n_clicks=0),
                    html.Div(id='fraud-output'),
                ]),
            ]),
        ])

    return html.Div()

# Callback for Credit Prediction with API error handling and retry
@app.callback(
    Output('credit-output', 'children'),
    Input('credit-predict-button', 'n_clicks'),
    State('credit-input', 'value')
)
def update_credit_output(n_clicks, input_value):
    global credit_features
    if not input_value or n_clicks == 0:
        return ""
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            if not credit_features:
                credit_features, _ = get_api_features()
            data = json.loads(input_value)
            missing = [f for f in credit_features if f not in data]
            if missing:
                return f"Error: Missing features: {missing}. Required: {credit_features}"
            response = requests.post('http://localhost:5000/predict_credit', json=data, timeout=5)
            response.raise_for_status()
            result = response.json()
            if 'error' in result:
                return f"Error: {result['error']}"
            return f"Prediction: {result['prediction']}"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "Error: API server not running. Please start the API on localhost:5000 (run 'python serve_model.py' in the api/ directory)."
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to connect to API - {str(e)}"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format. Please check your input."
        except Exception as e:
            return f"Error: Unexpected error - {str(e)}"

# Callback for Fraud Prediction with API error handling and retry
@app.callback(
    Output('fraud-output', 'children'),
    Input('fraud-predict-button', 'n_clicks'),
    State('fraud-input', 'value')
)
def update_fraud_output(n_clicks, input_value):
    global fraud_features
    if not input_value or n_clicks == 0:
        return ""
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            if not fraud_features:
                _, fraud_features = get_api_features()
            data = json.loads(input_value)
            missing = [f for f in fraud_features if f not in data]
            if missing:
                return f"Error: Missing features: {missing}. Required: {fraud_features}"
            response = requests.post('http://localhost:5000/predict_fraud', json=data, timeout=5)
            response.raise_for_status()
            result = response.json()
            if 'error' in result:
                return f"Error: {result['error']}"
            return f"Prediction: {result['prediction']}"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "Error: API server not running. Please start the API on localhost:5000 (run 'python serve_model.py' in the api/ directory)."
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to connect to API - {str(e)}"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format. Please check your input."
        except Exception as e:
            return f"Error: Unexpected error - {str(e)}"

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)