# layout/risk_forecast.py

import os
import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
from xgboost import Booster, DMatrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# === Load XGBoost Booster model ===
current_dir = os.path.dirname(os.path.abspath(__file__))
# From dashboard/layout/ to notebooks/
model_path = os.path.join(current_dir, '..', '..', 'notebooks', 'xgb_model.json')
model = Booster()
model.load_model(model_path)
print(f"✅ Risk Analysis: Model loaded from: {model_path}")

# === Load encoders ===
# From dashboard/layout/ to notebooks/
encoder_dir = os.path.join(current_dir, '..', '..', 'notebooks')
le_target = joblib.load(os.path.join(encoder_dir, 'label_encoder_target.pkl'))
le_condition = joblib.load(os.path.join(encoder_dir, 'label_encoder_condition.pkl'))
le_day_of_week = joblib.load(os.path.join(encoder_dir, 'label_encoder_day_of_week.pkl'))
le_season = joblib.load(os.path.join(encoder_dir, 'label_encoder_season.pkl'))
le_part_of_day = joblib.load(os.path.join(encoder_dir, 'label_encoder_part_of_day.pkl'))
print(f"✅ Risk Analysis: Encoders loaded from: {encoder_dir}")

# === Load training data for statistical averages ===
try:
    # Corrected path: From dashboard/layout/ to dashboard/data/
    training_data_path = os.path.join(current_dir, '..', 'data', 'cleaned_dataset.xlsx')
    TRAINING_DATA = pd.read_excel(training_data_path)
    
    # Remove rows with missing weather data (same as your training script)
    columns_with_nulls = ["temp_c", "humidity", "wind_kph", "precip_mm", "cloud", "pressure_mb", "condition"]
    TRAINING_DATA = TRAINING_DATA.dropna(subset=columns_with_nulls)
    
    print("✅ Risk Analysis: Loaded training data for risk forecasting")
    print(f"📊 Training data shape: {TRAINING_DATA.shape}")
    print(f"📁 Data loaded from: {training_data_path}")
    
except Exception as e:
    print(f"❌ Risk Analysis: Could not load training data: {e}")
    print(f"🔍 Tried to load from: {training_data_path}")
    print(f"🔍 Current directory: {current_dir}")
    print(f"🔍 Check if file exists: {os.path.exists(training_data_path) if 'training_data_path' in locals() else 'Path not defined'}")
    TRAINING_DATA = None

# === Feature definitions - CORRECTED ORDER TO MATCH MODEL ===
NUMERICAL_FEATURES = ['acci_x', 'acci_y', 'acci_hour', 'temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb']
CATEGORICAL_FEATURES = ['condition', 'day_of_week', 'season', 'part_of_day']  # Fixed order
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# === Class labels and colors ===
severity_classes = ['minor', 'severe', 'simple']
severity_color_map = {'minor': '#ff7f0e', 'severe': '#d62728', 'simple': '#1f77b4'}

def get_season_from_date(date_obj):
    """Get season from date"""
    month = date_obj.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def get_part_of_day_distribution():
    """Get typical distribution of accidents by part of day"""
    if TRAINING_DATA is not None:
        return TRAINING_DATA['part_of_day'].value_counts(normalize=True).to_dict()
    else:
        return {'Morning': 0.2, 'Afternoon': 0.35, 'Evening': 0.25, 'Night': 0.2}

def get_statistical_averages_for_period(target_date, location=None):
    """
    Get statistical averages for weather and accident patterns for a specific date/period
    
    Args:
        target_date: datetime.date object
        location: dict with 'lat' and 'lon' keys (optional)
    
    Returns:
        dict: Complete feature set with statistical averages
    """
    if TRAINING_DATA is None:
        # Fallback defaults if no training data
        return {
            'acci_x': 25.2048, 'acci_y': 55.2708, 'acci_hour': 14,
            'temp_c': 25.0, 'humidity': 60, 'wind_kph': 15,
            'precip_mm': 2, 'cloud': 40, 'pressure_mb': 1013,
            'condition': 'Clear', 'day_of_week': target_date.strftime('%A'),
            'season': get_season_from_date(target_date), 'part_of_day': 'Afternoon'
        }
    
    # Get season and day of week from target date
    season = get_season_from_date(target_date)
    day_of_week = target_date.strftime('%A')
    month = target_date.month
    
    print(f"🗓️ Calculating averages for: {target_date} (Season: {season}, Day: {day_of_week})")
    
    # Filter data by season and month for more accurate historical averages
    seasonal_data = TRAINING_DATA[TRAINING_DATA['season'] == season].copy()
    
    # If we have enough data, also filter by month
    monthly_data = TRAINING_DATA[TRAINING_DATA.index.to_series().apply(
        lambda x: pd.to_datetime(TRAINING_DATA.loc[x, 'acci_date'] if 'acci_date' in TRAINING_DATA.columns else '2023-01-01').month == month
    ) if 'acci_date' in TRAINING_DATA.columns else seasonal_data.index]
    
    # Use monthly data if available, otherwise seasonal
    if len(monthly_data) >= 50:  # Minimum threshold for monthly averages
        reference_data = monthly_data
        print(f"📊 Using monthly averages ({len(reference_data)} samples)")
    else:
        reference_data = seasonal_data
        print(f"📊 Using seasonal averages ({len(reference_data)} samples)")
    
    # Calculate statistical averages
    averages = {}
    
    # Numerical features - use median for robustness
    for feature in NUMERICAL_FEATURES:
        if feature in ['acci_x', 'acci_y']:
            # Location handling
            if location:
                averages[feature] = location['lat' if feature == 'acci_x' else 'lon']
            else:
                # Use overall center point or median of accidents
                averages[feature] = reference_data[feature].median()
        elif feature == 'acci_hour':
            # Peak accident hour for the season/month
            hour_distribution = reference_data[feature].value_counts()
            averages[feature] = hour_distribution.index[0]  # Most common hour
        else:
            # Weather features
            averages[feature] = reference_data[feature].median()
    
    # Categorical features - use mode (most common)
    averages['day_of_week'] = day_of_week
    averages['season'] = season
    
    # Weather condition - most common for the period
    condition_mode = reference_data['condition'].mode()
    averages['condition'] = condition_mode.iloc[0] if len(condition_mode) > 0 else 'Clear'
    
    # Part of day - most common accident time for the period
    part_of_day_mode = reference_data['part_of_day'].mode()
    averages['part_of_day'] = part_of_day_mode.iloc[0] if len(part_of_day_mode) > 0 else 'Afternoon'
    
    print(f"✅ Generated statistical averages: {averages}")
    return averages

def prepare_model_input_for_forecast(features_dict):
    """
    Prepare input for XGBoost model with proper encoding and CORRECT FEATURE ORDER
    """
    # Create DataFrame with EXACT feature order that model expects
    input_df = pd.DataFrame([features_dict])
    
    # Encode categorical features
    try:
        input_df['condition'] = le_condition.transform([features_dict['condition']])[0]
        input_df['day_of_week'] = le_day_of_week.transform([features_dict['day_of_week']])[0]
        input_df['season'] = le_season.transform([features_dict['season']])[0]
        input_df['part_of_day'] = le_part_of_day.transform([features_dict['part_of_day']])[0]
    except ValueError as e:
        print(f"❌ Encoding error: {e}")
        raise
    
    # Ensure correct data types
    for feature in NUMERICAL_FEATURES:
        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce')
    
    # CRITICAL: Reorder columns to match model's expected feature order
    expected_order = ['acci_x', 'acci_y', 'acci_hour', 'temp_c', 'humidity', 'wind_kph', 
                      'precip_mm', 'cloud', 'pressure_mb', 'condition', 'day_of_week', 'season', 'part_of_day']
    
    input_df = input_df[expected_order]
    
    print(f"🔧 Model input shape: {input_df.shape}")
    print(f"🔧 Model input columns: {list(input_df.columns)}")
    print(f"🔧 Model input dtypes: {input_df.dtypes.to_dict()}")
    
    return input_df

def get_historical_severity_distribution(target_date):
    """Get historical severity distribution for the target period"""
    if TRAINING_DATA is None:
        return {'simple': 0.6, 'minor': 0.3, 'severe': 0.1}
    
    season = get_season_from_date(target_date)
    day_of_week = target_date.strftime('%A')
    
    # Filter by season and day of week
    filtered_data = TRAINING_DATA[
        (TRAINING_DATA['season'] == season) & 
        (TRAINING_DATA['day_of_week'] == day_of_week)
    ]
    
    if len(filtered_data) < 10:  # Fallback to season only
        filtered_data = TRAINING_DATA[TRAINING_DATA['season'] == season]
    
    if len(filtered_data) > 0:
        severity_dist = filtered_data['severity'].value_counts(normalize=True).to_dict()
        # Ensure all classes are present
        for severity in severity_classes:
            if severity not in severity_dist:
                severity_dist[severity] = 0.0
        return severity_dist
    else:
        return {'simple': 0.6, 'minor': 0.3, 'severe': 0.1}

def get_layout():
    return dbc.Container([
        # Title
        html.H2("Strategic Planning", className="mb-4", style={'text-align': 'left'}),
        
        # How to Use Section - Full Width Rectangle
        dbc.Card([
            dbc.CardHeader(html.H5("📋 How to Use", className="mb-0")),
            dbc.CardBody([
                html.P([
                    "This strategic risk forecasting tool helps you plan for future periods by analyzing historical patterns. ",
                    "Select any future date to see predicted risk levels based on seasonal trends, weather patterns, and historical accident data. ",
                    "Optionally specify coordinates for location-specific analysis. Perfect for answering questions like ",
                    html.Strong("'How risky will next December be?'"), " or ", html.Strong("'Should we increase patrols during summer weekends?'")
                ], className="mb-2"),
                html.P([
                    html.Strong("🎯 Key Features: "), 
                    "Long-term planning • Historical pattern analysis • Weather-based predictions • Location-specific insights"
                ], className="mb-0", style={'color': '#0066cc'})
            ])
        ], className="mb-4", color="light"),
        
        # Input Controls Row - All 4 items side by side
        dbc.Row([
            dbc.Col([
                dbc.Label("📅 Future Date", className="fw-bold mb-2"),
                dcc.DatePickerSingle(
                    id='risk-date-picker',
                    min_date_allowed=datetime.date.today(),
                    max_date_allowed=datetime.date.today() + datetime.timedelta(days=365*10),
                    date=datetime.date.today() + datetime.timedelta(days=30),
                    display_format='DD/MM/YYYY',
                    style={'width': '100%'}
                )
            ], md=3),
            
            dbc.Col([
                dbc.Label("🌍 Latitude", className="fw-bold mb-2"),
                dbc.Input(
                    id="forecast-lat", 
                    type="number", 
                    step="any",
                    min=-90, 
                    max=90, 
                    placeholder="Optional",
                    style={'width': '100%'}
                )
            ], md=3),
            
            dbc.Col([
                dbc.Label("🌍 Longitude", className="fw-bold mb-2"),
                dbc.Input(
                    id="forecast-lon", 
                    type="number", 
                    step="any",
                    min=-180, 
                    max=180, 
                    placeholder="Optional",
                    style={'width': '100%'}
                )
            ], md=3),
            
            dbc.Col([
                dbc.Label("⚡ Action", className="fw-bold mb-2"),
                dbc.Button(
                    "🔮 Generate Strategic Forecast", 
                    id='run-risk-btn', 
                    color="primary", 
                    size="lg",
                    style={'width': '100%', 'height': '38px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                )
            ], md=3)
        ], className="mb-4"),
        
        # Status and Results
        html.Div(id='forecast-status', className="mb-3"),
        
        # Charts Row - Side by Side
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='risk-prediction-graph', style={'height': '500px'})
            ], md=6),
            
            dbc.Col([
                dcc.Graph(id='risk-comparison-graph', style={'height': '500px'})
            ], md=6)
        ], className="mb-4"),
        
        # Additional Information Row
        dbc.Row([
            dbc.Col([
                html.Div(id='forecast-summary')
            ], md=6),
            
            dbc.Col([
                html.Div(id='historical-context')
            ], md=6)
        ], className="mb-4"),
        
        # Results and Features (Full Width)
        html.Div(id='risk-prediction-output', className="mb-4"),
        html.Div(id='forecast-features', className="mb-4")
        
    ], fluid=True)

def register_callbacks(app):
    @app.callback(
        [Output('forecast-status', 'children'),
         Output('risk-prediction-output', 'children'),
         Output('risk-prediction-graph', 'figure'),
         Output('risk-comparison-graph', 'figure'),
         Output('forecast-summary', 'children'),
         Output('historical-context', 'children'),
         Output('forecast-features', 'children')],
        [Input('run-risk-btn', 'n_clicks')],
        [State('risk-date-picker', 'date'),
         State('forecast-lat', 'value'),
         State('forecast-lon', 'value')]
    )
    def update_risk_forecast(n_clicks, selected_date, lat, lon):
        if not n_clicks:
            empty_fig = px.bar(title="Select a date and click 'Generate Strategic Risk Forecast'")
            empty_fig.update_layout(height=500)
            return "", html.Div(), empty_fig, empty_fig, html.Div(), html.Div(), html.Div()

        try:
            # Convert selected_date to datetime.date
            target_date = pd.to_datetime(selected_date).date()
            today = datetime.date.today()
            days_ahead = (target_date - today).days
            
            print(f"🎯 Forecasting for: {target_date} ({days_ahead} days ahead)")
            
            # Prepare location if provided
            location = None
            if lat is not None and lon is not None:
                location = {'lat': lat, 'lon': lon}
                location_info = f"📍 Location: {lat:.4f}, {lon:.4f}"
            else:
                location_info = "🌍 Regional average (no specific location)"
            
            # Status information
            status = dbc.Alert([
                html.P(f"📅 Target Date: {target_date.strftime('%A, %B %d, %Y')} | ⏰ {days_ahead:,} days ahead | {location_info}"),
                html.P("📊 Analysis complete - using historical statistical averages for prediction", className="mb-0")
            ], color="success")
            
            # Get statistical averages for the target period
            statistical_features = get_statistical_averages_for_period(target_date, location)
            
            # Prepare model input and make prediction
            model_input_df = prepare_model_input_for_forecast(statistical_features)
            
            # Make prediction using XGBoost
            dinput = DMatrix(model_input_df)
            pred_probs = model.predict(dinput)[0]
            predicted_class = np.argmax(pred_probs)
            predicted_label = severity_classes[predicted_class]
            confidence = pred_probs[predicted_class]
            
            print(f"✅ Strategic forecast: {predicted_label} (confidence: {confidence:.3f})")
            
            # Get historical severity distribution for comparison
            historical_dist = get_historical_severity_distribution(target_date)
            
            # Create prediction results display
            results = dbc.Card([
                dbc.CardHeader(html.H4("🎯 Strategic Risk Forecast Results")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H2(f"EXPECTED: {predicted_label.upper()}", 
                                   style={'color': severity_color_map[predicted_label], 'text-align': 'center', 'margin-bottom': '10px'}),
                            html.P(f"Model Confidence: {confidence:.1%}", 
                                  style={'text-align': 'center', 'font-size': '1.2em', 'margin-bottom': '20px'})
                        ], md=6),
                        dbc.Col([
                            html.P("📊 Risk Probability Distribution:", style={'font-weight': 'bold', 'margin-bottom': '10px'}),
                            html.Ul([
                                html.Li(f"{cls.capitalize()}: {prob:.1%}")
                                for cls, prob in zip(severity_classes, pred_probs)
                            ], style={'margin-bottom': '0'})
                        ], md=6)
                    ])
                ])
            ], color="primary", outline=True)
            
            # Create probability bar chart
            prob_fig = px.bar(
                x=severity_classes, y=pred_probs,
                labels={'x': 'Severity Level', 'y': 'Predicted Probability'},
                color=severity_classes,
                color_discrete_map=severity_color_map,
                title=f"Risk Forecast for {target_date.strftime('%B %Y')}"
            )
            prob_fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            prob_fig.update_layout(showlegend=False, height=500)
            
            # Create comparison chart (Predicted vs Historical)
            comparison_data = pd.DataFrame({
                'Severity': severity_classes * 2,
                'Probability': list(pred_probs) + [historical_dist.get(s, 0) for s in severity_classes],
                'Type': ['Predicted'] * 3 + ['Historical Average'] * 3
            })
            
            comparison_fig = px.bar(
                comparison_data, x='Severity', y='Probability', color='Type',
                barmode='group', 
                labels={'Probability': 'Probability', 'Severity': 'Severity Level'},
                title=f"Predicted vs Historical Risk Distribution",
                color_discrete_map={'Predicted': '#2E86AB', 'Historical Average': '#A23B72'}
            )
            comparison_fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            comparison_fig.update_layout(height=500)
            
            # Summary insights
            season = get_season_from_date(target_date)
            day_name = target_date.strftime('%A')
            
            # Calculate risk level
            severe_risk = pred_probs[severity_classes.index('severe')]
            if severe_risk > 0.2:
                risk_level = "🔴 HIGH RISK"
                risk_color = "danger"
            elif severe_risk > 0.1:
                risk_level = "🟡 MODERATE RISK"
                risk_color = "warning"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "success"
            
            summary = dbc.Card([
                dbc.CardHeader(html.H5("📋 Strategic Planning Summary")),
                dbc.CardBody([
                    html.H4(risk_level, style={'text-align': 'center', 'margin-bottom': '15px'}),
                    html.Hr(),
                    html.P(f"🌍 Season: {season} | 📅 Day Type: {day_name}"),
                    html.P(f"🌡️ Expected Weather: {statistical_features['condition']} | 🕐 Peak Risk Time: {statistical_features['part_of_day']}"),
                    html.Hr(),
                    html.P("💡 Strategic Recommendations:", style={'font-weight': 'bold'}),
                    html.Ul([
                        html.Li("Increase patrol presence during peak risk hours") if severe_risk > 0.15 else html.Li("Standard patrol schedule recommended"),
                        html.Li("Weather-specific safety campaigns") if statistical_features['condition'] != 'Clear' else html.Li("General safety awareness sufficient"),
                        html.Li("Enhanced emergency response readiness") if severe_risk > 0.2 else html.Li("Standard emergency response protocol")
                    ])
                ])
            ], color=risk_color, outline=True)
            
            # Historical context
            if TRAINING_DATA is not None:
                sample_size = len(TRAINING_DATA[
                    (TRAINING_DATA['season'] == season) & 
                    (TRAINING_DATA['day_of_week'] == day_name)
                ])
                
                context = dbc.Card([
                    dbc.CardHeader(html.H5("📚 Historical Context")),
                    dbc.CardBody([
                        html.P(f"📊 Analysis based on {sample_size:,} historical accidents"),
                        html.P(f"🗓️ Period: {season} {day_name}s"),
                        html.P("📈 Historical Severity Distribution:", style={'font-weight': 'bold'}),
                        html.Ul([
                            html.Li(f"{severity.capitalize()}: {prob:.1%}")
                            for severity, prob in historical_dist.items()
                        ])
                    ])
                ], color="secondary", outline=True)
            else:
                context = dbc.Alert("⚠️ Limited historical data available. Predictions based on general patterns.", color="warning")
            
            # CREATE DETAILED FEATURES DISPLAY
            features_display = dbc.Card([
                dbc.CardHeader(html.H5("🔧 Model Input Features")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P("📊 Numerical Features:", style={'font-weight': 'bold'}),
                            html.Ul([
                                html.Li(f"Location: ({statistical_features['acci_x']:.4f}, {statistical_features['acci_y']:.4f})"),
                                html.Li(f"Hour: {statistical_features['acci_hour']}:00"),
                                html.Li(f"Temperature: {statistical_features['temp_c']:.1f}°C"),
                                html.Li(f"Humidity: {statistical_features['humidity']:.0f}%"),
                                html.Li(f"Wind Speed: {statistical_features['wind_kph']:.1f} kph"),
                                html.Li(f"Precipitation: {statistical_features['precip_mm']:.1f} mm"),
                                html.Li(f"Cloud Cover: {statistical_features['cloud']:.0f}%"),
                                html.Li(f"Pressure: {statistical_features['pressure_mb']:.0f} mb")
                            ])
                        ], md=6),
                        dbc.Col([
                            html.P("📋 Categorical Features:", style={'font-weight': 'bold'}),
                            html.Ul([
                                html.Li(f"Weather Condition: {statistical_features['condition']}"),
                                html.Li(f"Day of Week: {statistical_features['day_of_week']}"),
                                html.Li(f"Season: {statistical_features['season']}"),
                                html.Li(f"Part of Day: {statistical_features['part_of_day']}")
                            ])
                        ], md=6)
                    ])
                ])
            ], color="info", outline=True)
            
            return status, results, prob_fig, comparison_fig, summary, context, features_display
            
        except Exception as e:
            print(f"❌ Risk forecast error: {e}")
            import traceback
            traceback.print_exc()
            
            error_fig = px.bar(title=f"❌ Error: {str(e)}")
            error_fig.update_layout(height=500)
            error_msg = dbc.Alert(f"Error during risk forecasting: {str(e)}", color="danger")
            
            return error_msg, html.Div(), error_fig, error_fig, html.Div(), html.Div(), html.Div()