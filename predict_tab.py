# layout/predict_tab.py

import os
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from xgboost import Booster, DMatrix
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# === Load XGBoost Booster model ===
current_dir = os.path.dirname(os.path.abspath(__file__))
# From dashboard/layout/ to notebooks/
model_path = os.path.join(current_dir, '..', '..', 'notebooks', 'xgb_model.json')
model = Booster()
model.load_model(model_path)
print(f" Model loaded from: {model_path}")

# === Load encoders ===
# From dashboard/layout/ to notebooks/
encoder_dir = os.path.join(current_dir, '..', '..', 'notebooks')
le_condition = joblib.load(os.path.join(encoder_dir, 'label_encoder_condition.pkl'))
le_day_of_week = joblib.load(os.path.join(encoder_dir, 'label_encoder_day_of_week.pkl'))
le_season = joblib.load(os.path.join(encoder_dir, 'label_encoder_season.pkl'))
le_part_of_day = joblib.load(os.path.join(encoder_dir, 'label_encoder_part_of_day.pkl'))
print(f" Encoders loaded from: {encoder_dir}")

# === Load training data for context-aware defaults ===
try:
    # Corrected path: From dashboard/layout/ to dashboard/data/
    training_data_path = os.path.join(current_dir, '..', 'data', 'cleaned_dataset.xlsx')
    TRAINING_DATA = pd.read_excel(training_data_path)
    
    # Remove rows with missing weather data (same as your training script)
    columns_with_nulls = ["temp_c", "humidity", "wind_kph", "precip_mm", "cloud", "pressure_mb", "condition"]
    TRAINING_DATA = TRAINING_DATA.dropna(subset=columns_with_nulls)
    
    print(" Loaded training data for context-aware defaults")
    print(f" Training data shape: {TRAINING_DATA.shape}")
    print(f" Data loaded from: {training_data_path}")
    
except Exception as e:
    print(f" Could not load training data: {e}")
    print(f" Tried to load from: {training_data_path}")
    print(f" Current directory: {current_dir}")
    print(f" Check if file exists: {os.path.exists(training_data_path) if 'training_data_path' in locals() else 'Path not defined'}")
    TRAINING_DATA = None

# === Feature definitions ===
NUMERICAL_FEATURES = ['acci_x', 'acci_y', 'acci_hour', 'temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb']
CATEGORICAL_FEATURES = ['condition', 'day_of_week', 'season', 'part_of_day']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# === Class labels and colors ===
severity_classes = ['minor', 'severe', 'simple']
severity_color_map = {'minor': '#ff7f0e', 'severe': '#d62728', 'simple': '#1f77b4'}

def get_context_aware_defaults(user_input, training_data):
    """
    Get realistic defaults based on user's partial input using progressive filtering
    
    Args:
        user_input: dict of {feature: value} that user has provided
        training_data: pandas DataFrame of training data
    
    Returns:
        tuple: (defaults_dict, num_samples_used, context_description)
    """
    if training_data is None:
        # Fallback to basic defaults if no training data
        fallback_defaults = {
            'acci_x': 25.2048, 'acci_y': 55.2708, 'acci_hour': 12,
            'temp_c': 25.0, 'humidity': 50.0, 'wind_kph': 10.0,
            'precip_mm': 0.0, 'cloud': 30.0, 'pressure_mb': 1013.0,
            'condition': 'Clear', 'day_of_week': 'Monday', 
            'season': 'Summer', 'part_of_day': 'Afternoon'
        }
        return fallback_defaults, 0, ["No training data available"]
    
    # Progressive filtering strategy: Apply constraints in order of importance
    # Priority: categorical features first, then numerical
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
        if feature in user_input and user_input[feature] is not None:
            previous_size = len(current_data)
            
            if constraint_type == 'categorical':
                # Apply categorical constraint
                temp_data = current_data[current_data[feature] == user_input[feature]]
                constraint_desc = f"{feature}={user_input[feature]}"
                
            else:  # numerical
                # Apply numerical constraint with tolerance
                std_val = training_data[feature].std()
                tolerance = max(std_val * 0.3, training_data[feature].quantile(0.75) - training_data[feature].quantile(0.25)) / 4
                
                temp_data = current_data[
                    (current_data[feature] >= user_input[feature] - tolerance) & 
                    (current_data[feature] <= user_input[feature] + tolerance)
                ]
                constraint_desc = f"{feature}≈{user_input[feature]}"
            
            # Only apply constraint if we still have reasonable data
            if len(temp_data) >= 1:  # Keep even 1 sample if it matches our constraints!
                current_data = temp_data
                applied_constraints.append(constraint_desc)
                print(f"Applied {constraint_desc}: {previous_size} → {len(current_data)} samples")
            else:
                print(f"Skipped {constraint_desc}: would result in 0 samples")
                # Don't apply this constraint, keep previous data
    
    print(f"Final context: {len(current_data)} samples with constraints: {applied_constraints}")
    
    # Generate defaults from the filtered data
    defaults = {}
    
    for feature in ALL_FEATURES:
        if feature not in user_input or user_input[feature] is None:
            if feature in current_data.columns and len(current_data) > 0:
                if feature in CATEGORICAL_FEATURES:
                    # Most common value in filtered context
                    mode_values = current_data[feature].mode()
                    if len(mode_values) > 0:
                        defaults[feature] = mode_values.iloc[0]
                    else:
                        # Fallback to global mode if no mode in filtered data
                        defaults[feature] = training_data[feature].mode().iloc[0]
                else:
                    # Median value in filtered context
                    defaults[feature] = current_data[feature].median()
            else:
                # Ultimate fallback defaults
                fallback_defaults = {
                    'acci_x': 25.2048, 'acci_y': 55.2708, 'acci_hour': 12,
                    'temp_c': 25.0, 'humidity': 50.0, 'wind_kph': 10.0,
                    'precip_mm': 0.0, 'cloud': 30.0, 'pressure_mb': 1013.0,
                    'condition': 'Clear', 'day_of_week': 'Monday', 
                    'season': 'Summer', 'part_of_day': 'Afternoon'
                }
                defaults[feature] = fallback_defaults[feature]
    
    # Create context description
    if applied_constraints:
        context_desc = applied_constraints
    else:
        context_desc = ["Global statistics (no specific constraints)"]
    
    return defaults, len(current_data), context_desc

def prepare_model_input(user_input, defaults):
    """
    Prepare input for XGBoost model with proper encoding
    """
    # Combine user input with defaults
    complete_input = {}
    for feature in ALL_FEATURES:
        if feature in user_input and user_input[feature] is not None:
            complete_input[feature] = user_input[feature]
        else:
            complete_input[feature] = defaults[feature]
    
    # Create DataFrame
    input_df = pd.DataFrame([complete_input])
    
    # Encode categorical features
    try:
        input_df['condition'] = le_condition.transform([complete_input['condition']])[0]
        input_df['day_of_week'] = le_day_of_week.transform([complete_input['day_of_week']])[0]
        input_df['season'] = le_season.transform([complete_input['season']])[0]
        input_df['part_of_day'] = le_part_of_day.transform([complete_input['part_of_day']])[0]
    except ValueError as e:
        print(f"Encoding error: {e}")
        raise
    
    # Ensure correct data types
    for feature in NUMERICAL_FEATURES:
        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce')
    
    return input_df, complete_input

def get_matching_locations_and_count(user_input, predicted_severity, training_data, max_locations=100):
    """
    Get historical accident locations that match user input and predicted severity
    Also returns the exact count for accurate reporting
    
    Args:
        user_input: dict of user provided features
        predicted_severity: predicted severity class ('simple', 'minor', 'severe')
        training_data: pandas DataFrame of training data
        max_locations: maximum number of locations to return
    
    Returns:
        tuple: (locations_df, total_matching_count)
    """
    if training_data is None or 'severity' not in training_data.columns:
        return pd.DataFrame(), 0
    
    # Start with accidents of the predicted severity
    filtered_data = training_data[training_data['severity'] == predicted_severity].copy()
    
    # Apply user constraints to find similar accidents
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
    
    # Apply constraints progressively
    for feature, constraint_type in constraint_priority:
        if feature in user_input and user_input[feature] is not None:
            if constraint_type == 'categorical':
                # Apply categorical constraint
                temp_data = filtered_data[filtered_data[feature] == user_input[feature]]
                
            else:  # numerical
                # Apply numerical constraint with tolerance
                if feature not in ['acci_x', 'acci_y']:
                    std_val = training_data[feature].std()
                    tolerance = max(std_val * 0.3, training_data[feature].quantile(0.75) - training_data[feature].quantile(0.25)) / 4
                    
                    temp_data = filtered_data[
                        (filtered_data[feature] >= user_input[feature] - tolerance) & 
                        (filtered_data[feature] <= user_input[feature] + tolerance)
                    ]
                else:
                    # Skip coordinate constraints for location matching
                    continue
            
            # Only apply constraint if we still have reasonable data
            if len(temp_data) >= 1:
                filtered_data = temp_data
    
    # Store total count before location filtering
    total_matching_count = len(filtered_data)
    
    # Ensure we have location data for the map
    filtered_data = filtered_data.dropna(subset=['acci_x', 'acci_y'])
    
    # Limit the number of locations to avoid overcrowding the map
    if len(filtered_data) > max_locations:
        filtered_data = filtered_data.sample(n=max_locations, random_state=42)
    
    locations_df = filtered_data[['acci_x', 'acci_y', 'severity']].rename(columns={'acci_x': 'lat', 'acci_y': 'lon'})
    
    return locations_df, total_matching_count

def get_layout():
    """
    Returns the layout for the prediction tab with reorganized structure
    """
    return dbc.Container([
        # Title
        html.H2("Predict Severity", className="mb-3", style={'font-weight': 'bold'}),
        
        # Full Width Instructions Box
        dbc.Card([
            dbc.CardBody([
                html.H6("How to Use This Dashboard", className="card-title", style={'font-weight': 'bold'}),
                html.P([
                    "1. Fill in the available accident and weather information below", html.Br(),
                    "2. Leave fields empty if data is not available - the system will use intelligent defaults", html.Br(),
                    "3. Click 'Predict Severity' to get the accident severity prediction", html.Br(),
                    "4. View the results in the bar chart and location map below"
                ], className="card-text", style={'font-size': '0.9em', 'margin-bottom': '0'})
            ])
        ], className="mb-4", style={'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'}),
        
        # Input Section
        html.H5("Input Data", className="mb-3", style={'font-weight': 'bold'}),

        # First row: Latitude, Hour, Day of Week, Season, Part of Day, Weather Condition
        dbc.Row([
            dbc.Col([
                dbc.Label("Latitude", className="mb-1"),
                dbc.Input(
                    id="input-acci_x", 
                    type="number", 
                    step="any", 
                    placeholder="e.g., 25.2048",
                    className="mb-3"
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Hour (0-23)", className="mb-1"),
                dbc.Input(
                    id="input-acci_hour", 
                    type="number", 
                    min=0, 
                    max=23, 
                    step=1, 
                    placeholder="e.g., 14",
                    className="mb-3"
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Day of Week", className="mb-1"),
                dcc.Dropdown(
                    id="input-day_of_week", 
                    options=[{'label': d, 'value': d} for d in le_day_of_week.classes_],
                    value=None, 
                    clearable=True,
                    placeholder="Select day",
                    className="mb-3"
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Season", className="mb-1"),
                dcc.Dropdown(
                    id="input-season", 
                    options=[{'label': s, 'value': s} for s in le_season.classes_],
                    value=None, 
                    clearable=True,
                    placeholder="Select season",
                    className="mb-3"
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Part of Day", className="mb-1"),
                dcc.Dropdown(
                    id="input-part_of_day", 
                    options=[{'label': p, 'value': p} for p in le_part_of_day.classes_],
                    value=None, 
                    clearable=True,
                    placeholder="Select time",
                    className="mb-3"
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Weather Condition", className="mb-1"),
                dcc.Dropdown(
                    id="input-condition", 
                    options=[{'label': c, 'value': c} for c in le_condition.classes_],
                    value=None, 
                    clearable=True,
                    placeholder="Select condition",
                    className="mb-3"
                ),
            ], width=2),
        ], className="mb-3"),

        # Unified layout row: Left = stacked inputs, Right = bar chart + map
        dbc.Row([
            # LEFT COLUMN: All weather inputs stacked vertically
            dbc.Col([
                dbc.Label("Longitude", className="mb-1"),
                dbc.Input(id="input-acci_y", type="number", step="any", placeholder="e.g., 55.2708", className="mb-3"),

                dbc.Label("Temperature (°C)", className="mb-1"),
                dbc.Input(id="input-temp_c", type="number", step=0.1, placeholder="e.g., 35", className="mb-3"),

                dbc.Label("Humidity (%)", className="mb-1"),
                dbc.Input(id="input-humidity", type="number", min=0, max=100, step=1, placeholder="e.g., 65", className="mb-3"),

                dbc.Label("Wind Speed (kph)", className="mb-1"),
                dbc.Input(id="input-wind_kph", type="number", min=0, step=0.1, placeholder="e.g., 15", className="mb-3"),

                dbc.Label("Precipitation (mm)", className="mb-1"),
                dbc.Input(id="input-precip_mm", type="number", min=0, step=0.1, placeholder="e.g., 0", className="mb-3"),

                dbc.Label("Cloud Cover (%)", className="mb-1"),
                dbc.Input(id="input-cloud", type="number", min=0, max=100, step=1, placeholder="e.g., 25", className="mb-3"),

                dbc.Label("Pressure (mb)", className="mb-1"),
                dbc.Input(id="input-pressure_mb", type="number", min=900, max=1100, step=0.1, placeholder="e.g., 1013", className="mb-3"),

                dbc.Button("Predict Severity", id="btn-predict", color="primary", size="lg", className="w-100 mb-4", style={'font-weight': 'bold'}),
            ], width=2),

            # RIGHT COLUMN: Bar chart + map stacked
            dbc.Col([
                html.Div(id="prediction-results", className="mb-3"),

                dcc.Graph(id='bar-prediction-probabilities',
                        style={'height': '350px', 'margin': '0', 'padding': '0', 'marginBottom': '24px'},
                        config={'displayModeBar': False}),
                
                html.Div(id="context-info", className="mb-3"),

                html.H5("Location Analysis", className="mb-3", style={'font-weight': 'bold'}),
                dcc.Graph(id='map-predicted-location',
                        style={'height': '400px', 'margin': '0', 'padding': '0'},
                        config={'displayModeBar': True}),
            ], width=10),
        ], className="mb-4"),
        
    ], fluid=True, className="p-4 prediction-tab-container")

def register_callbacks(app):
    """
    Register callbacks for the prediction tab
    """
    @app.callback(
        [Output('bar-prediction-probabilities', 'figure'),
         Output('map-predicted-location', 'figure'),
         Output('prediction-results', 'children'),
         Output('context-info', 'children')],
        [Input('btn-predict', 'n_clicks')],
        [State(f'input-{field}', 'value') for field in ALL_FEATURES]
    )
    def predict_with_context(n_clicks, *input_values):
        if not n_clicks:
            # Return empty figures
            empty_fig = px.bar(title="Click 'Predict Severity' to see results")
            empty_fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False
            )
            empty_map = px.scatter_mapbox(
                pd.DataFrame({'lat': [25.2048], 'lon': [55.2708]}),
                lat='lat', lon='lon', zoom=10, height=400
            )
            empty_map.update_layout(
                mapbox_style="open-street-map", 
                margin=dict(l=0, r=0, t=0, b=0)
            )
            return empty_fig, empty_map, html.Div(), html.Div()

        try:
            # Parse user input
            user_input = {}
            for i, feature in enumerate(ALL_FEATURES):
                if input_values[i] is not None:
                    user_input[feature] = input_values[i]
            
            # Check if at least one feature is provided
            if not user_input:
                error_fig = px.bar(title="Please enter at least one feature")
                error_fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )
                error_map = px.scatter_mapbox(pd.DataFrame({'lat': [0], 'lon': [0]}), lat='lat', lon='lon', zoom=1, height=400)
                error_map.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                return (
                    error_fig,
                    error_map,
                    dbc.Alert("Please enter at least one feature to make a prediction.", color="warning"),
                    html.Div()
                )
            
            print(f"User input: {user_input}")
            
            # Check if user provided coordinates
            user_provided_location = ('acci_x' in user_input and user_input['acci_x'] is not None and 
                                    'acci_y' in user_input and user_input['acci_y'] is not None)
            
            # Get context-aware defaults using progressive filtering
            defaults, n_similar_for_defaults, context_info = get_context_aware_defaults(user_input, TRAINING_DATA)
            
            # Prepare model input
            model_input_df, complete_input = prepare_model_input(user_input, defaults)
            
            print(f"Complete input: {complete_input}")
            
            # Make prediction using XGBoost
            dinput = DMatrix(model_input_df)
            pred_probs = model.predict(dinput)[0]
            predicted_class = np.argmax(pred_probs)
            predicted_label = severity_classes[predicted_class]
            confidence = pred_probs[predicted_class]
            
            print(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
            
            # Get accurate count of similar accidents with the predicted severity
            if not user_provided_location:
                _, actual_similar_count = get_matching_locations_and_count(user_input, predicted_label, TRAINING_DATA)
            else:
                actual_similar_count = n_similar_for_defaults
            
            # Create results display
            results = dbc.Card([
                dbc.CardHeader([
                    html.H5("Prediction Results", className="mb-0"),
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H3(f"{predicted_label.upper()}", 
                                   style={'color': severity_color_map[predicted_label], 'text-align': 'center', 'margin-bottom': '5px'}),
                            html.P(f"Confidence: {confidence:.1%}", style={'text-align': 'center', 'font-size': '1.1em', 'margin-bottom': '10px'}),
                        ], width=6),
                        dbc.Col([
                            html.P("Probability Distribution:", style={'font-weight': 'bold', 'margin-bottom': '5px', 'font-size': '0.9em'}),
                            html.Ul([
                                html.Li(f"{cls.capitalize()}: {prob:.1%}", style={'font-size': '0.85em'})
                                for cls, prob in zip(severity_classes, pred_probs)
                            ], style={'margin-bottom': '0'})
                        ], width=6)
                    ])
                ])
            ], color="success", outline=True, style={'margin-bottom': '10px'})
            
            # Determine context quality
            if actual_similar_count >= 100:
                context_quality = "🟢 Excellent"
                context_color = "success"
            elif actual_similar_count >= 20:
                context_quality = "🟡 Good"  
                context_color = "warning"
            elif actual_similar_count >= 5:
                context_quality = "🟠 Limited but Relevant"
                context_color = "warning"
            elif actual_similar_count >= 1:
                context_quality = "🔴 Very Limited but Specific"
                context_color = "danger"
            else:
                context_quality = "⚪ Global Fallback"
                context_color = "secondary"
            
            # Create location map
            if user_provided_location:
                # Show single location provided by user
                location_data = pd.DataFrame({
                    'lat': [complete_input['acci_x']], 
                    'lon': [complete_input['acci_y']], 
                    'severity': [predicted_label],
                    'confidence': [confidence],
                    'type': ['User Location']
                })
                
                map_fig = px.scatter_mapbox(
                    location_data,
                    lat='lat', lon='lon',
                    color='severity',
                    color_discrete_map=severity_color_map,
                    size=[25],
                    zoom=12, height=400,
                    hover_data={'confidence': ':.1%', 'type': True},
                    title=f"Your Location - Predicted: {predicted_label.upper()}"
                )
                
                map_info = f"Showing your specific location with predicted severity: {predicted_label.upper()}"
                
            else:
                # Show multiple historical locations with matching criteria and predicted severity
                matching_locations, total_matching_count = get_matching_locations_and_count(user_input, predicted_label, TRAINING_DATA)
                
                if len(matching_locations) > 0:
                    # Add some variety in marker sizes for better visualization
                    marker_sizes = [8] * len(matching_locations)
                    
                    map_fig = px.scatter_mapbox(
                        matching_locations,
                        lat='lat', lon='lon',
                        color='severity',
                        color_discrete_map=severity_color_map,
                        size=marker_sizes,
                        zoom=10, height=400,
                        hover_data={'severity': True},
                        title=f"Historical {predicted_label.upper()} Accidents Matching Your Criteria"
                    )
                    
                    if len(matching_locations) < total_matching_count:
                        map_info = f"Showing {len(matching_locations)} of {total_matching_count} historical {predicted_label} accidents that match your criteria"
                    else:
                        map_info = f"Showing all {total_matching_count} historical {predicted_label} accidents that match your criteria"
                    
                else:
                    # Fallback to default location if no matching data found
                    location_data = pd.DataFrame({
                        'lat': [25.2048], 
                        'lon': [55.2708], 
                    })
                    
                    map_fig = px.scatter_mapbox(
                        location_data,
                        lat='lat', lon='lon',
                        zoom=10, height=400,
                        title="No Historical Matching Locations Found"
                    )
                    
                    map_info = "No historical accidents found matching your criteria and predicted severity"
            
            map_fig.update_layout(
                mapbox_style="open-street-map", 
                margin={"r": 0, "t": 30, "l": 0, "b": 0}
            )
            
            # Create context information display
            context_display = dbc.Card([
                dbc.CardHeader(html.H6("Analysis Context", className="mb-0")),
                dbc.CardBody([
                    html.P([
                        f"Found ", html.Strong(f"{actual_similar_count:,}"), f" {predicted_label} accidents matching your criteria"
                    ], className="mb-2"),
                    html.P([
                        "Context Quality: ", html.Strong(context_quality)
                    ], className="mb-2"),
                    html.P(f"Applied Constraints: {', '.join(context_info)}", className="mb-2"),
                    html.Hr(),
                    html.P(map_info, style={'color': 'blue', 'font-weight': 'bold', 'font-size': '0.9em'}, className="mb-0"),
                ])
            ], color=context_color, outline=True)
            
            # Create probability bar chart
            bar_fig = px.bar(
                x=severity_classes, y=pred_probs,
                labels={'x': 'Severity Level', 'y': 'Probability'},
                color=severity_classes,
                color_discrete_map=severity_color_map,
                title=f"Predicted Severity: {predicted_label.upper()} ({confidence:.1%} confidence)"
            )
            bar_fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            bar_fig.update_layout(showlegend=False, height=400)
            
            return bar_fig, map_fig, results, context_display
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            error_fig = px.bar(title=f"Error: {str(e)}")
            error_map = px.scatter_mapbox(
                pd.DataFrame({'lat': [0], 'lon': [0]}),
                lat='lat', lon='lon', zoom=1, height=400
            )
            error_map.update_layout(mapbox_style="open-street-map")
            
            error_msg = dbc.Alert(f"Error during prediction: {str(e)}", color="danger")
            
            return error_fig, error_map, error_msg, html.Div()
