# layout/future_forecast.py

import os
import requests
from datetime import datetime, timedelta
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from xgboost import Booster, DMatrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# === API Configuration ===
WEATHERAPI_KEY = "c62d9e545d084bfd86c03743252906"  

# === Load XGBoost Booster model ===
current_dir = os.path.dirname(os.path.abspath(__file__))
# From dashboard/layout/ to notebooks/
model_path = os.path.join(current_dir, '..', '..', 'notebooks', 'xgb_model.json')
model = Booster()
model.load_model(model_path)
print(f"✅ Future Forecast: Model loaded from: {model_path}")

# === Load encoders ===
# From dashboard/layout/ to notebooks/
encoder_dir = os.path.join(current_dir, '..', '..', 'notebooks')
le_target = joblib.load(os.path.join(encoder_dir, 'label_encoder_target.pkl'))
le_condition = joblib.load(os.path.join(encoder_dir, 'label_encoder_condition.pkl'))
le_day_of_week = joblib.load(os.path.join(encoder_dir, 'label_encoder_day_of_week.pkl'))
le_season = joblib.load(os.path.join(encoder_dir, 'label_encoder_season.pkl'))
le_part_of_day = joblib.load(os.path.join(encoder_dir, 'label_encoder_part_of_day.pkl'))
print(f"✅ Future Forecast: Encoders loaded from: {encoder_dir}")

# === Load training data for context-aware defaults ===
try:
    # Corrected path: From dashboard/layout/ to dashboard/data/
    training_data_path = os.path.join(current_dir, '..', 'data', 'cleaned_dataset.xlsx')
    TRAINING_DATA = pd.read_excel(training_data_path)
    
    # Remove rows with missing weather data (same as your training script)
    columns_with_nulls = ["temp_c", "humidity", "wind_kph", "precip_mm", "cloud", "pressure_mb", "condition"]
    TRAINING_DATA = TRAINING_DATA.dropna(subset=columns_with_nulls)
    
    print("✅ Future Forecast: Loaded training data for context-aware defaults")
    print(f"📊 Training data shape: {TRAINING_DATA.shape}")
    print(f"📁 Data loaded from: {training_data_path}")
    
except Exception as e:
    print(f"❌ Future Forecast: Could not load training data: {e}")
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

def map_weather_condition(weather_api_condition):
    """
    Map WeatherAPI.com condition text to your model's condition categories
    """
    condition_lower = weather_api_condition.lower()
    
    # Get available conditions from the encoder
    available_conditions = le_condition.classes_
    
    # Map based on common weather conditions - adjust based on your model's categories
    if any(word in condition_lower for word in ['clear', 'sunny']):
        return 'Clear' if 'Clear' in available_conditions else available_conditions[0]
    elif any(word in condition_lower for word in ['cloud', 'overcast']):
        return 'Cloudy' if 'Cloudy' in available_conditions else available_conditions[0]
    elif any(word in condition_lower for word in ['rain', 'drizzle', 'shower']):
        return 'Rainy' if 'Rainy' in available_conditions else available_conditions[0]
    elif any(word in condition_lower for word in ['snow', 'blizzard']):
        return 'Snow' if 'Snow' in available_conditions else available_conditions[0]
    elif any(word in condition_lower for word in ['fog', 'mist']):
        return 'Fog' if 'Fog' in available_conditions else available_conditions[0]
    elif any(word in condition_lower for word in ['thunder', 'storm']):
        return 'Thunderstorm' if 'Thunderstorm' in available_conditions else available_conditions[0]
    else:
        # Default to first available condition (usually most common)
        return available_conditions[0]

def get_season_from_date(date_str):
    """
    Determine season from date string (YYYY-MM-DD)
    """
    dt = datetime.fromisoformat(date_str)
    month = dt.month
    
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Autumn'

def get_day_of_week_from_date(date_str):
    """
    Get day of week from date string (YYYY-MM-DD)
    """
    dt = datetime.fromisoformat(date_str)
    return dt.strftime('%A')  # Returns full day name like 'Monday'

def get_part_of_day_from_hour(hour):
    """
    Convert hour (0-23) to part of day category
    """
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:  # 0-5
        return 'Night'

def get_context_aware_defaults(user_input, forecast_data, training_data):
    """
    Enhanced version that combines user input, forecast data, and progressive filtering
    """
    if training_data is None:
        # Fallback defaults
        fallback_defaults = {
            'acci_x': 25.2048, 'acci_y': 55.2708, 'acci_hour': 12,
            'temp_c': 25.0, 'humidity': 50.0, 'wind_kph': 10.0,
            'precip_mm': 0.0, 'cloud': 30.0, 'pressure_mb': 1013.0,
            'condition': le_condition.classes_[0], 'day_of_week': 'Monday', 
            'season': 'Summer', 'part_of_day': 'Afternoon'
        }
        return fallback_defaults, 0, ["No training data available"]
    
    # Combine forecast data with user input (user input takes priority)
    combined_input = forecast_data.copy()
    combined_input.update(user_input)
    
    # Progressive filtering strategy
    constraint_priority = [
        ('condition', 'categorical'),
        ('season', 'categorical'), 
        ('part_of_day', 'categorical'),
        ('day_of_week', 'categorical'),
        ('temp_c', 'numerical'),
        ('humidity', 'numerical'),
        ('wind_kph', 'numerical'),
        ('precip_mm', 'numerical'),
        ('cloud', 'numerical'),
        ('pressure_mb', 'numerical'),
        ('acci_hour', 'numerical'),
        ('acci_x', 'numerical'),
        ('acci_y', 'numerical')
    ]
    
    # Start with full training data
    current_data = training_data.copy()
    applied_constraints = []
    
    # Apply constraints progressively
    for feature, constraint_type in constraint_priority:
        if feature in combined_input and combined_input[feature] is not None:
            previous_size = len(current_data)
            
            if constraint_type == 'categorical':
                # Apply categorical constraint (exact match)
                temp_data = current_data[current_data[feature] == combined_input[feature]]
                constraint_desc = f"{feature}={combined_input[feature]}"
                
            else:  # numerical
                # Apply numerical constraint with tolerance
                if len(current_data) > 0 and feature in current_data.columns:
                    std_val = current_data[feature].std()
                    if pd.isna(std_val) or std_val == 0:
                        std_val = training_data[feature].std()
                    
                    tolerance = max(std_val * 0.3, 
                                  (training_data[feature].quantile(0.75) - training_data[feature].quantile(0.25)) / 4)
                    
                    temp_data = current_data[
                        (current_data[feature] >= combined_input[feature] - tolerance) & 
                        (current_data[feature] <= combined_input[feature] + tolerance)
                    ]
                    constraint_desc = f"{feature}≈{combined_input[feature]}"
                else:
                    temp_data = current_data
                    constraint_desc = f"{feature} (skipped - no data)"
            
            # Only apply constraint if we still have reasonable data
            if len(temp_data) >= 1:
                current_data = temp_data
                applied_constraints.append(constraint_desc)
                print(f"✅ Applied {constraint_desc}: {previous_size} → {len(current_data)} samples")
            else:
                print(f"⚠️ Skipped {constraint_desc}: would result in 0 samples")
    
    print(f"🎯 Final context: {len(current_data)} samples with constraints: {applied_constraints}")
    
    # Generate defaults from the filtered data
    defaults = {}
    
    for feature in ALL_FEATURES:
        # Use combined input first, then filtered context defaults
        if feature in combined_input and combined_input[feature] is not None:
            defaults[feature] = combined_input[feature]
        elif feature in current_data.columns and len(current_data) > 0:
            if feature in CATEGORICAL_FEATURES:
                # Most common value in filtered context
                mode_values = current_data[feature].mode()
                if len(mode_values) > 0:
                    defaults[feature] = mode_values.iloc[0]
                else:
                    # Fallback to global mode if no mode in filtered data
                    defaults[feature] = training_data[feature].mode().iloc[0]
            else:
                # Median value in filtered context (robust against outliers)
                defaults[feature] = current_data[feature].median()
        else:
            # Ultimate fallback defaults
            fallback_defaults = {
                'acci_x': 25.2048, 'acci_y': 55.2708, 'acci_hour': 12,
                'temp_c': 25.0, 'humidity': 50.0, 'wind_kph': 10.0,
                'precip_mm': 0.0, 'cloud': 30.0, 'pressure_mb': 1013.0,
                'condition': le_condition.classes_[0], 'day_of_week': 'Monday', 
                'season': 'Summer', 'part_of_day': 'Afternoon'
            }
            defaults[feature] = fallback_defaults[feature]
    
    # Create context description
    if applied_constraints:
        context_desc = applied_constraints
    else:
        context_desc = ["Global statistics (no specific constraints)"]
    
    return defaults, len(current_data), context_desc

def prepare_model_input(complete_input):
    """
    Prepare input for XGBoost model with proper encoding and CORRECT FEATURE ORDER
    """
    # Create DataFrame with EXACT feature order that model expects
    input_df = pd.DataFrame([complete_input])
    
    # Encode categorical features
    try:
        input_df['condition'] = le_condition.transform([complete_input['condition']])[0]
        input_df['day_of_week'] = le_day_of_week.transform([complete_input['day_of_week']])[0]
        input_df['season'] = le_season.transform([complete_input['season']])[0]
        input_df['part_of_day'] = le_part_of_day.transform([complete_input['part_of_day']])[0]
    except ValueError as e:
        print(f"❌ Encoding error: {e}")
        print(f"Available conditions: {le_condition.classes_}")
        print(f"Available days: {le_day_of_week.classes_}")
        print(f"Available seasons: {le_season.classes_}")
        print(f"Available parts: {le_part_of_day.classes_}")
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

def get_weather_forecast(city, date, hour):
    """
    Fetch weather forecast from WeatherAPI.com with improved error handling
    """
    if WEATHERAPI_KEY == "your_weatherapi_key_here" or WEATHERAPI_KEY == "cb8abdc4079f47ce878135604252103":
        # Mock forecast data for demo purposes
        print("🔧 Using mock weather data (API key not configured)")
        return {
            'temp_c': 28.5,
            'humidity': 65,
            'wind_kph': 12.3,
            'precip_mm': 0.0,
            'cloud': 25,
            'pressure_mb': 1013.2,
            'condition': 'Clear',
            'lat': 25.2048,
            'lon': 55.2708,
            'location_name': city
        }
    
    try:
        # Validate date is not too far in future (WeatherAPI supports up to 14 days)
        target_date = datetime.fromisoformat(date)
        days_ahead = (target_date - datetime.now()).days
        
        if days_ahead > 14:
            print(f"⚠️ Date too far ahead ({days_ahead} days), using historical average")
            return None
        
        # Call WeatherAPI.com forecast endpoint
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={city}&dt={date}&aqi=no&alerts=no"
        response = requests.get(url, timeout=10)
        
        print(f"🌐 WeatherAPI request: {url}")
        print(f"🌐 Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Weather API error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        # Validate response structure
        if 'forecast' not in data or 'forecastday' not in data['forecast']:
            print(f"❌ Invalid response structure: {data}")
            return None
        
        # Find forecast for chosen hour
        forecast_hours = data['forecast']['forecastday'][0]['hour']
        forecast_for_hour = next((h for h in forecast_hours if h['time'].endswith(f"{hour:02d}:00")), None)
        
        if not forecast_for_hour:
            print(f"❌ No forecast data for hour {hour}")
            # Use the closest available hour
            forecast_for_hour = forecast_hours[min(hour, len(forecast_hours)-1)]
            print(f"🔧 Using closest hour: {forecast_for_hour['time']}")
        
        # Extract relevant features
        weather_data = {
            'temp_c': forecast_for_hour['temp_c'],
            'humidity': forecast_for_hour['humidity'],
            'wind_kph': forecast_for_hour['wind_kph'],
            'precip_mm': forecast_for_hour['precip_mm'],
            'cloud': forecast_for_hour['cloud'],
            'pressure_mb': forecast_for_hour['pressure_mb'],
            'condition': map_weather_condition(forecast_for_hour['condition']['text']),
            'lat': data['location']['lat'],
            'lon': data['location']['lon'],
            'location_name': data['location']['name']
        }
        
        print(f"✅ Weather data retrieved: {weather_data}")
        return weather_data
        
    except requests.exceptions.Timeout:
        print("❌ Weather API timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Weather API request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching weather data: {e}")
        return None

def get_layout():
    """
    Returns the layout for the future forecast tab with reorganized structure
    """
    return dbc.Container([

        # Title
        html.H2("Future Forecast", className="mb-3", style={'font-weight': 'bold'}),

        # Full Width Instructions Box
        dbc.Card([
            dbc.CardBody([
                html.H6("How to Use This Dashboard", className="card-title", style={'font-weight': 'bold'}),
                html.P([
                    "1. Enter the location (city), date, and time for your forecast prediction", html.Br(),
                    "2. Fill in any additional weather or location data if available", html.Br(),
                    "3. Leave fields empty if data is not available - the system will use intelligent forecast data", html.Br(),
                    "4. Click 'Get Future Forecast' to fetch weather data and predict accident severity", html.Br(),
                    "5. View the prediction results in the bar chart and location map below"
                ], className="card-text", style={'font-size': '0.9em', 'margin-bottom': '0'})
            ])
        ], className="mb-4", style={'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'}),

        # Input Section Header
        html.H5("Input Data", className="mb-3", style={'font-weight': 'bold'}),

        # First row: Horizontal inputs (Latitude, City, Date, Time, Weather Condition)
        dbc.Row([
            dbc.Col([
                dbc.Label("Latitude", className="mb-1"),
                dbc.Input(id="override-acci_x", type="number", step="any", placeholder="e.g., 25.2048", className="mb-3"),
            ], width=2),

            dbc.Col([
                dbc.Label("City/Place", className="mb-1"),
                dbc.Input(id='city-input', placeholder='e.g., Dubai, New York, London', type='text', value='Dubai', className="mb-3")
            ], width=2),

            dbc.Col([
                dbc.Label("Date", className="mb-1"),
                dcc.DatePickerSingle(
                    id='date-picker', 
                    date=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    min_date_allowed=datetime.now().date(),
                    max_date_allowed=(datetime.now() + timedelta(days=14)).date(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'},
                    className="mb-3"
                )
            ], width=2),

            dbc.Col([
                dbc.Label("Time (24h)", className="mb-1"),
                dcc.Dropdown(
                    id='time-picker',
                    options=[{'label': f"{h:02d}:00", 'value': h} for h in range(24)],
                    value=15,  # 3 PM
                    clearable=False,
                    placeholder="Select time",
                    className="mb-3"
                ),
            ], width=2),

            dbc.Col([
                dbc.Label("Weather Condition Override", className="mb-1"),
                dcc.Dropdown(
                    id="override-condition", 
                    options=[{'label': c, 'value': c} for c in le_condition.classes_],
                    value=None, 
                    clearable=True,
                    placeholder="Override forecast condition",
                    className="mb-3"
                ),
            ], width=4),
        ], className="mb-3"),

        # Combined second row: Inputs (left), Outputs (right)
        dbc.Row([
            # LEFT COLUMN: all inputs stacked vertically
            dbc.Col([
                dbc.Label("Longitude", className="mb-1"),
                dbc.Input(id="override-acci_y", type="number", step="any", placeholder="e.g., 55.2708", className="mb-3"),

                dbc.Label("Temperature (°C)", className="mb-1"),
                dbc.Input(id="override-temp_c", type="number", step=0.1, placeholder="Override forecast temp", className="mb-3"),

                dbc.Label("Humidity (%)", className="mb-1"),
                dbc.Input(id="override-humidity", type="number", min=0, max=100, step=1, placeholder="Override forecast humidity", className="mb-3"),

                dbc.Label("Wind Speed (kph)", className="mb-1"),
                dbc.Input(id="override-wind_kph", type="number", min=0, step=0.1, placeholder="Override forecast wind", className="mb-3"),

                dbc.Label("Precipitation (mm)", className="mb-1"),
                dbc.Input(id="override-precip_mm", type="number", min=0, step=0.1, placeholder="Override forecast rain", className="mb-3"),

                dbc.Label("Cloud Cover (%)", className="mb-1"),
                dbc.Input(id="override-cloud", type="number", min=0, max=100, step=1, placeholder="Override forecast clouds", className="mb-3"),

                dbc.Label("Pressure (mb)", className="mb-1"),
                dbc.Input(id="override-pressure_mb", type="number", min=900, max=1100, step=0.1, placeholder="Override forecast pressure", className="mb-3"),

                dbc.Label("Hour of Day (0–23)", className="mb-1"),
                dbc.Input(id="override-acci_hour", type="number", min=0, max=23, placeholder="Override hour of day", className="mb-3"),

                dbc.Button("Get Future Forecast", id='submit-forecast-button', color="primary", size="lg",
                           className="w-100 mb-4", style={'font-weight': 'bold'}),
            ], width=2),

            # RIGHT COLUMN: results, bar chart, context, map
            dbc.Col([
                # Forecast results
                html.Div(id="forecast-results", className="mb-3"),

                # Context information
                html.Div(id="forecast-context-info", className="mb-3"),

                # Prediction probability chart
                dcc.Loading(
                    dcc.Graph(
                        id='forecast-prediction-chart',
                        style={'height': '350px', 'margin': '0', 'padding': '0'},
                        config={'displayModeBar': True}
                    ),
                    type='default'
                ),

                html.H5("Location Analysis", className="mb-3", style={'font-weight': 'bold'}),
                
                # Location map
                dcc.Graph(
                    id='forecast-map-location',
                    style={'height': '400px', 'margin': '0', 'padding': '0'},
                    config={'displayModeBar': True}
                ),
                
            ], width=10),
        ], className="mb-4"),

    ], fluid=True, className="p-4")
def register_callbacks(app):
    """
    Register callbacks for the future forecast tab
    """
    @app.callback(
        [Output('forecast-prediction-chart', 'figure'),
         Output('forecast-map-location', 'figure'),
         Output('forecast-results', 'children'),
         Output('forecast-context-info', 'children')],
        [Input('submit-forecast-button', 'n_clicks')],
        [State('city-input', 'value'),
         State('date-picker', 'date'),
         State('time-picker', 'value')] +
        [State(f'override-{field}', 'value') for field in NUMERICAL_FEATURES] +
        [State('override-condition', 'value')]
    )
    def predict_with_forecast(n_clicks, city, date, hour, *override_values):
        if not n_clicks:
            # Return empty figures
            empty_fig = px.bar(title="Enter location, date, time and click 'Get Forecast & Predict'")
            empty_map = px.scatter_mapbox(
                pd.DataFrame({'lat': [25.2048], 'lon': [55.2708]}),
                lat='lat', lon='lon', zoom=10, height=400
            )
            empty_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            return empty_fig, empty_map, html.Div(), html.Div()

        try:
            # Validate inputs
            if not city or not date or hour is None:
                error_msg = dbc.Alert("Please enter city, date, and time.", color="warning")
                empty_fig = px.bar(title="❌ Missing required inputs")
                empty_map = px.scatter_mapbox(pd.DataFrame({'lat': [0], 'lon': [0]}), lat='lat', lon='lon', zoom=1, height=400)
                empty_map.update_layout(mapbox_style="open-street-map")
                return empty_fig, empty_map, error_msg, html.Div()

            print(f"🌤️ Fetching forecast for {city} on {date} at {hour}:00")
            
            # Get weather forecast
            weather_data = get_weather_forecast(city, date, hour)
            if not weather_data:
                error_msg = dbc.Alert("Failed to fetch weather forecast. Please check city name and date (max 14 days ahead).", color="danger")
                empty_fig = px.bar(title="❌ Weather API Error")
                empty_map = px.scatter_mapbox(pd.DataFrame({'lat': [0], 'lon': [0]}), lat='lat', lon='lon', zoom=1, height=400)
                empty_map.update_layout(mapbox_style="open-street-map")
                return empty_fig, empty_map, error_msg, html.Div()

            # Parse user overrides
            user_overrides = {}
            
            # Numerical overrides
            for i, feature in enumerate(NUMERICAL_FEATURES):
                if override_values[i] is not None:
                    user_overrides[feature] = override_values[i]
            
            # Condition override
            condition_override = override_values[-1]
            if condition_override:
                user_overrides['condition'] = condition_override

            print(f"🎛️ User overrides: {user_overrides}")

            # Create forecast data with time context
            forecast_data = {
                'acci_x': weather_data['lat'],
                'acci_y': weather_data['lon'],
                'acci_hour': hour,
                'temp_c': weather_data['temp_c'],
                'humidity': weather_data['humidity'],
                'wind_kph': weather_data['wind_kph'],
                'precip_mm': weather_data['precip_mm'],
                'cloud': weather_data['cloud'],
                'pressure_mb': weather_data['pressure_mb'],
                'condition': weather_data['condition'],
                'day_of_week': get_day_of_week_from_date(date),
                'season': get_season_from_date(date),
                'part_of_day': get_part_of_day_from_hour(hour)
            }
            
            print(f"🌤️ Forecast data: {forecast_data}")

            # Get context-aware defaults using progressive filtering
            defaults, n_similar, context_info = get_context_aware_defaults(user_overrides, forecast_data, TRAINING_DATA)
            
            print(f"⚙️ Final defaults: {defaults}")

            # Prepare model input
            model_input_df = prepare_model_input(defaults)
            
            # Make prediction using XGBoost
            dinput = DMatrix(model_input_df)
            pred_probs = model.predict(dinput)[0]
            predicted_class = np.argmax(pred_probs)
            predicted_label = severity_classes[predicted_class]
            confidence = pred_probs[predicted_class]
            
            print(f"✅ Prediction: {predicted_label} (confidence: {confidence:.3f})")

            # FIXED: Determine data sources for display and create final display values
            forecast_used = []
            user_overridden = []
            context_filled = []
            
            # Create display values that reflect what's actually being used
            display_weather_data = {}
            final_location = {'lat': defaults['acci_x'], 'lon': defaults['acci_y']}
            
            for feature in ALL_FEATURES:
                if feature in user_overrides:
                    user_overridden.append(feature)
                    # Use overridden value for display
                    if feature in ['temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb', 'condition']:
                        display_weather_data[feature] = user_overrides[feature]
                elif feature in forecast_data:
                    forecast_used.append(feature)
                    # Use forecast value for display
                    if feature in ['temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb', 'condition']:
                        display_weather_data[feature] = forecast_data[feature]
                else:
                    context_filled.append(feature)
                    # Use context-filled value for display
                    if feature in ['temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb', 'condition']:
                        display_weather_data[feature] = defaults[feature]

            # FIXED: Use actual final location for display
            final_location_name = weather_data.get('location_name', city)
            if ('acci_x' in user_overrides and user_overrides['acci_x'] is not None) or ('acci_y' in user_overrides and user_overrides['acci_y'] is not None):
    # User overrode location - only show the single prediction point
                final_location_name = f"Custom Location ({final_location['lat']:.4f}, {final_location['lon']:.4f})"

            # Create results display with CORRECTED weather info
            results = dbc.Card([
                dbc.CardHeader(html.H4("🎯 Future Forecast Prediction Results")),
                dbc.CardBody([
                    html.H2(f"{predicted_label.upper()}", 
                        style={'color': severity_color_map[predicted_label], 'text-align': 'center'}),
                    html.P(f"Confidence: {confidence:.1%}", style={'text-align': 'center', 'font-size': '1.2em'}),
                    html.Hr(),
                    html.P([
                        "📅 ", html.Strong(f"{date} at {hour:02d}:00"), " in ", 
                        html.Strong(final_location_name)
                    ]),
                    html.P([
                        "🌍 Coordinates: ", 
                        html.Strong(f"({final_location['lat']:.4f}, {final_location['lon']:.4f})")
                    ]),
                    html.Hr(),
                    
                    # Weather information display (FIXED to show actual used values)
                    html.H5("🌤️ Weather Conditions Used for Prediction:"),
                    dbc.Row([
                        dbc.Col([
                            html.P([html.Strong("Temperature: "), f"{display_weather_data.get('temp_c', 'N/A')}°C"]),
                            html.P([html.Strong("Humidity: "), f"{display_weather_data.get('humidity', 'N/A')}%"]),
                            html.P([html.Strong("Wind Speed: "), f"{display_weather_data.get('wind_kph', 'N/A')} kph"]),
                            html.P([html.Strong("Condition: "), f"{display_weather_data.get('condition', 'N/A')}"]),
                        ], width=6),
                        dbc.Col([
                            html.P([html.Strong("Precipitation: "), f"{display_weather_data.get('precip_mm', 'N/A')} mm"]),
                            html.P([html.Strong("Cloud Cover: "), f"{display_weather_data.get('cloud', 'N/A')}%"]),
                            html.P([html.Strong("Pressure: "), f"{display_weather_data.get('pressure_mb', 'N/A')} mb"]),
                        ], width=6),
                    ]),
                    
                    # Data source information
                    html.Hr(),
                    html.H6("📊 Data Sources:"),
                    html.Ul([
                        html.Li([html.Strong("From Weather API: "), ", ".join(forecast_used) if forecast_used else "None"]),
                        html.Li([html.Strong("User Overridden: "), ", ".join(user_overridden) if user_overridden else "None"]),
                        html.Li([html.Strong("Context Filled: "), ", ".join(context_filled) if context_filled else "None"]),
                    ])
                ])
            ])

            # Create context information display
            context_info_display = dbc.Card([
                dbc.CardHeader(html.H5("🧠 Progressive Context Analysis")),
                dbc.CardBody([
                    html.P([
                        html.Strong(f"Similar Historical Cases: "), f"{n_similar:,} records found"
                    ]),
                    html.P([html.Strong("Applied Constraints:")]),
                    html.Ul([html.Li(constraint) for constraint in context_info]),
                    html.P([
                        html.Strong("Context Quality: "),
                        "Excellent" if n_similar > 100 else "Good" if n_similar > 50 else "Limited" if n_similar > 10 else "Very Limited"
                    ])
                ])
            ])

            # Create prediction probability chart
            prob_df = pd.DataFrame({
                'Severity': severity_classes,
                'Probability': pred_probs,
                'Color': [severity_color_map[s] for s in severity_classes]
            })
            
            prediction_chart = px.bar(
                prob_df, x='Severity', y='Probability',
                color='Severity',
                color_discrete_map=severity_color_map,
                title=f"🎯 Accident Severity Prediction for {final_location_name}",
                labels={'Probability': 'Prediction Probability'},
                text='Probability'
            )
            prediction_chart.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            prediction_chart.update_layout(
                showlegend=False,
                yaxis_tickformat='.0%',
                height=400
            )

            # FIXED: Create map with correct location (use final_location instead of weather_data)
            # Create map with prediction location and similar historical locations
            map_data = []

            # Add the prediction location (main point)
            if 'acci_x' in user_overrides or 'acci_y' in user_overrides:
                # User overrode location - only show the single prediction point
                map_data.append({
                    'lat': final_location['lat'],
                    'lon': final_location['lon'],
                    'location': final_location_name,
                    'prediction': predicted_label,
                    'confidence': f"{confidence:.1%}",
                    'type': 'Prediction Location',
                    'size': 15
                })
                map_title = f"📍 Custom Prediction Location: {final_location_name}"
            else:
                # Location not overridden - show prediction + similar historical locations
                map_data.append({
                    'lat': final_location['lat'],
                    'lon': final_location['lon'],
                    'location': final_location_name,
                    'prediction': predicted_label,
                    'confidence': f"{confidence:.1%}",
                    'type': 'Prediction Location',
                    'size': 20
                })
                
                # Get similar historical locations with same predicted severity
                similar_locations = get_similar_locations_with_severity(
                    defaults, predicted_label, TRAINING_DATA, max_locations=100
                )
                
                if not similar_locations.empty:
                    for _, row in similar_locations.iterrows():
                        map_data.append({
                            'lat': row['acci_x'],
                            'lon': row['acci_y'],
                            'location': f"Historical {predicted_label} accident",
                            'prediction': predicted_label,
                            'confidence': f"{row['similarity_score']:.1%} similar",
                            'type': 'Historical Location',
                            'size': 8
                        })
                    map_title = f"📍 {final_location_name} + {len(similar_locations)} Similar {predicted_label.title()} Locations"
                else:
                    map_title = f"📍 Prediction Location: {final_location_name}"

            map_df = pd.DataFrame(map_data)

            if not map_df.empty:
                location_map = px.scatter_mapbox(
                    map_df,
                    lat='lat',
                    lon='lon',
                    hover_name='location',
                    hover_data={
                        'prediction': True,
                        'confidence': True,
                        'type': True,
                        'lat': ':.4f',
                        'lon': ':.4f'
                    },
                    color='prediction',
                    color_discrete_map=severity_color_map,
                    size='size',
                    size_max=20,
                    zoom=10,
                    height=400,
                    title=map_title
                )
                location_map.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r": 0, "t": 30, "l": 0, "b": 0}
                )
            else:
                # Fallback empty map
                location_map = px.scatter_mapbox(
                    pd.DataFrame({'lat': [25.2048], 'lon': [55.2708]}),
                    lat='lat', lon='lon', zoom=10, height=400
                )
                location_map.update_layout(mapbox_style="open-street-map")
            
            # Return the successful results
            return prediction_chart, location_map, html.Div(), html.Div([results, html.Br(), context_info_display])

        except Exception as e:
            # Handle any errors that occur during processing
            print(f"❌ Error in forecast processing: {str(e)}")
            error_msg = dbc.Alert(f"An error occurred: {str(e)}", color="danger")
            empty_fig = px.bar(title="❌ Processing Error")
            empty_map = px.scatter_mapbox(pd.DataFrame({'lat': [0], 'lon': [0]}), lat='lat', lon='lon', zoom=1, height=400)
            empty_map.update_layout(mapbox_style="open-street-map")
            return empty_fig, empty_map, error_msg, html.Div()

# Additional helper function to fix the data override logic
def create_final_input_data(user_overrides, forecast_data, defaults):
    """
    Create final input data with proper priority: user_overrides > forecast_data > defaults
    """
    final_data = {}
    
    for feature in ALL_FEATURES:
        if feature in user_overrides and user_overrides[feature] is not None:
            # User override takes highest priority
            final_data[feature] = user_overrides[feature]
        elif feature in forecast_data and forecast_data[feature] is not None:
            # Forecast data takes second priority
            final_data[feature] = forecast_data[feature]
        else:
            # Context-aware defaults take lowest priority
            final_data[feature] = defaults[feature]
    
    return final_data

def get_similar_locations_with_severity(complete_input, predicted_severity, training_data, max_locations=50):
    """
    Optimized version: Find similar historical locations that match the predicted severity
    Uses efficient pandas operations instead of row-by-row iteration
    
    Args:
        complete_input: dict of complete input features
        predicted_severity: predicted severity class ('simple', 'minor', 'severe')
        training_data: pandas DataFrame of training data
        max_locations: maximum number of locations to return (reduced default)
    
    Returns:
        pandas DataFrame with similar locations
    """
    if training_data is None or 'severity' not in training_data.columns:
        return pd.DataFrame()
    
    try:
        # Start with accidents of the predicted severity
        severity_data = training_data[training_data['severity'] == predicted_severity].copy()
        
        if len(severity_data) == 0:
            return pd.DataFrame()
        
        # Apply user constraints to find similar accidents (same logic as context-aware defaults)
        constraint_priority = [
            ('condition', 'categorical'),
            ('season', 'categorical'), 
            ('part_of_day', 'categorical'),
            ('day_of_week', 'categorical'),
            ('temp_c', 'numerical'),
            ('humidity', 'numerical'),
            ('wind_kph', 'numerical'),
            ('precip_mm', 'numerical'),
            ('cloud', 'numerical'),
            ('pressure_mb', 'numerical'),
            ('acci_hour', 'numerical')
            # Exclude acci_x, acci_y from similarity matching to get diverse locations
        ]
        
        # Apply constraints progressively using efficient pandas operations
        filtered_data = severity_data.copy()
        
        for feature, constraint_type in constraint_priority:
            if feature in complete_input and complete_input[feature] is not None:
                if constraint_type == 'categorical':
                    # Apply categorical constraint (exact match)
                    temp_data = filtered_data[filtered_data[feature] == complete_input[feature]]
                    
                else:  # numerical
                    # Apply numerical constraint with tolerance
                    if len(filtered_data) > 0 and feature in filtered_data.columns:
                        std_val = training_data[feature].std()
                        if pd.isna(std_val) or std_val == 0:
                            std_val = training_data[feature].std()
                        
                        tolerance = max(std_val * 0.3, 
                                      (training_data[feature].quantile(0.75) - training_data[feature].quantile(0.25)) / 4)
                        
                        temp_data = filtered_data[
                            (filtered_data[feature] >= complete_input[feature] - tolerance) & 
                            (filtered_data[feature] <= complete_input[feature] + tolerance)
                        ]
                    else:
                        temp_data = filtered_data
                
                # Only apply constraint if we still have reasonable data
                if len(temp_data) >= 1:
                    filtered_data = temp_data
                
                # Early stopping if we have enough similar cases
                if len(filtered_data) <= max_locations * 2:
                    break
        
        # Ensure we have location data
        filtered_data = filtered_data.dropna(subset=['acci_x', 'acci_y'])
        
        if len(filtered_data) == 0:
            return pd.DataFrame()
        
        # Sample locations if we have too many (for performance)
        if len(filtered_data) > max_locations:
            filtered_data = filtered_data.sample(n=max_locations, random_state=42)
        
        # Create result DataFrame with required columns
        result_df = filtered_data[['acci_x', 'acci_y', 'severity']].copy()
        result_df['similarity_score'] = 1.0  # All are considered similar since they passed filters
        
        # Remove duplicate locations (within 0.01 degree tolerance) efficiently
        result_df = result_df.round({'acci_x': 2, 'acci_y': 2}).drop_duplicates(subset=['acci_x', 'acci_y'])
        
        print(f"✅ Found {len(result_df)} similar locations with {predicted_severity} severity")
        return result_df.head(max_locations)
        
    except Exception as e:
        print(f"❌ Error finding similar locations: {e}")
        return pd.DataFrame()


# Alternative even faster version using your working approach
def get_similar_locations_fast(complete_input, predicted_severity, training_data, max_locations=30):
    """
    Ultra-fast version based on your working get_matching_locations_and_count function
    """
    if training_data is None or 'severity' not in training_data.columns:
        return pd.DataFrame()
    
    # Use your proven efficient approach
    filtered_data = training_data[training_data['severity'] == predicted_severity].copy()
    
    # Apply only the most important constraints for performance
    priority_constraints = [
        ('condition', 'categorical'),
        ('season', 'categorical'), 
        ('part_of_day', 'categorical'),
        ('temp_c', 'numerical'),
        ('humidity', 'numerical')
    ]
    
    for feature, constraint_type in priority_constraints:
        if feature in complete_input and complete_input[feature] is not None:
            if constraint_type == 'categorical':
                temp_data = filtered_data[filtered_data[feature] == complete_input[feature]]
            else:  # numerical
                std_val = training_data[feature].std()
                tolerance = std_val * 0.5  # Wider tolerance for more results
                temp_data = filtered_data[
                    (filtered_data[feature] >= complete_input[feature] - tolerance) & 
                    (filtered_data[feature] <= complete_input[feature] + tolerance)
                ]
            
            if len(temp_data) >= 5:  # Keep reasonable minimum
                filtered_data = temp_data
    
    # Ensure location data and sample efficiently
    filtered_data = filtered_data.dropna(subset=['acci_x', 'acci_y'])
    
    if len(filtered_data) > max_locations:
        filtered_data = filtered_data.sample(n=max_locations, random_state=42)
    
    if len(filtered_data) == 0:
        return pd.DataFrame()
    
    result_df = filtered_data[['acci_x', 'acci_y', 'severity']].copy()
    result_df['similarity_score'] = 1.0
    
    return result_df

# Update the main prediction function to use the fixed logic
def update_prediction_logic():
    """
    This function shows the key changes needed in the main prediction callback.
    Replace the existing logic with this improved version.
    """
    # After getting user_overrides, forecast_data, and defaults, use:
    
    # Create final input data with proper priority
    final_input_data = create_final_input_data(user_overrides, forecast_data, defaults)
    
    # Update the location info
    final_location = {
        'lat': final_input_data['acci_x'], 
        'lon': final_input_data['acci_y']
    }
    
    # Update location name based on whether coordinates were overridden
    if 'acci_x' in user_overrides or 'acci_y' in user_overrides:
        final_location_name = f"Custom Location ({final_location['lat']:.4f}, {final_location['lon']:.4f})"
    else:
        final_location_name = weather_data.get('location_name', city)
    
    # Create display weather data using final values
    display_weather_data = {
        'temp_c': final_input_data['temp_c'],
        'humidity': final_input_data['humidity'],
        'wind_kph': final_input_data['wind_kph'],
        'precip_mm': final_input_data['precip_mm'],
        'cloud': final_input_data['cloud'],
        'pressure_mb': final_input_data['pressure_mb'],
        'condition': final_input_data['condition']
    }
    
    # Use final_input_data instead of defaults for model prediction
    model_input_df = prepare_model_input(final_input_data)
    
    # Continue with prediction as before...

print("✅ Future forecast code continuation and fixes applied!")
print("🔧 Key fixes implemented:")
print("   - Fixed location display to use actual coordinates (user override or API)")
print("   - Fixed weather data display to show actual values used in prediction")
print("   - Fixed map to show correct location")
print("   - Improved data priority logic: user_overrides > forecast_data > context_defaults")
print("   - Added proper data source tracking for transparency")