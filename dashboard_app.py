import dash
from dash import dcc, html, Input, Output, State
import dash.dash_table
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import dash_bootstrap_components as dbc
from dash.dependencies import ClientsideFunction

# === Load and Prepare Data ===
df = pd.read_excel('../data/cleaned_dataset.xlsx')
df.drop(columns=['acci_id', 'acci_ar', 'acci_en', 'description'], inplace=True)
df.dropna(subset=['acci_x', 'acci_y'], inplace=True)

if 'acci_hour' not in df.columns:
    df['acci_hour'] = pd.to_datetime(df['acci_time'], format='%H:%M:%S').dt.hour

df['acci_date'] = pd.to_datetime(df['acci_date'])  # If not already
df['acci_day'] = df['acci_date'].dt.day
df['acci_month'] = df['acci_date'].dt.month
df['year'] = df['acci_date'].dt.year

def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['part_of_day'] = df['acci_hour'].apply(get_part_of_day)
part_of_day_order = ['Morning', 'Afternoon', 'Evening', 'Night']
df['part_of_day'] = pd.Categorical(df['part_of_day'], categories=part_of_day_order, ordered=True)

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=weekday_order, ordered=True)

# KMeans clustering
coords = df[['acci_x', 'acci_y']]
kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
df['zone'] = kmeans.fit_predict(coords)

# === App Init ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Dubai Traffic Accidents Dashboard"

card_style = {
    'height': '25%',
    'width': '100%',
    'display': 'flex',
    'flexDirection': 'column',
    'justifyContent': 'center',
    'alignItems': 'center',
    'textAlign': 'center'
}

chart_style = {'height': '360px', 'width': '100%'}

month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# === Layout ===
app.layout = html.Div([
    # Title + Filters
    html.Div([
        html.H2("Dubai Traffic Accidents & Weather Conditions", style={'fontSize': '22px', 'margin': '0'}),
        html.Div([
            html.Div([
    html.Label("Day", style={'fontSize': '14px'}),
    dcc.Dropdown(
        options=[{'label': d, 'value': d} for d in sorted(df['acci_day'].dropna().unique())],
        id='day-filter', multi=True, placeholder="Select day(s)",
        style={'fontSize': '13px'}
    )
], style={'width': '16%', 'paddingRight': '10px'}),

html.Div([
    html.Label("Month", style={'fontSize': '14px'}),
    dcc.Dropdown(
        options=[{'label': month_names[m], 'value': m} for m in sorted(df['acci_month'].dropna().unique())],
        id='month-filter', multi=True, placeholder="Select month(s)",
        style={'fontSize': '13px'}
    )
], style={'width': '16%', 'paddingRight': '10px'}),

html.Div([
    html.Label("Year", style={'fontSize': '14px'}),
    dcc.Dropdown(
        options=[{'label': y, 'value': y} for y in sorted(df['year'].dropna().unique())],
        id='year-filter', multi=True, placeholder="Select year(s)",
        style={'fontSize': '13px'}
    )
], style={'width': '16%', 'paddingRight': '10px'}),

            html.Div([
                html.Label("Season", style={'fontSize': '14px'}),
                dcc.Dropdown(
                    options=[{'label': s, 'value': s} for s in df['season'].dropna().unique()],
                    id='season-filter', multi=True, placeholder="Select season(s)",
                    style={'fontSize': '13px'}
                )
            ], style={'width': '24%', 'paddingRight': '10px'}),

            html.Div([
                html.Label("Severity", style={'fontSize': '14px'}),
                dcc.Dropdown(
                    options=[{'label': s, 'value': s} for s in df['severity'].unique()],
                    id='severity-filter', multi=True, placeholder="Select severity level(s)",
                    style={'fontSize': '13px'}
                )
            ], style={'width': '24%', 'paddingRight': '10px'}),

            html.Div([
                html.Label("Hour Range", style={'fontSize': '14px'}),
                dcc.RangeSlider(
                    id='time-range-slider',
                    min=0, max=23, step=1,
                    value=[0, 23],
                    marks={i: f"{i}:00" for i in range(0, 24, 3)},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={'width': '28%'})
        ], style={'display': 'flex', 'paddingTop': '10px'})
    ], style={'padding': '10px 20px'}),

    # Top Row: Cards + Map + ML Placeholder
    html.Div([
        html.Div([
            dbc.Card([dbc.CardBody([html.H6("Total Accidents", className="text-center"), html.Div(id='total-accidents', className='card-metric text-center')])], className='mb-2', style=card_style),
            dbc.Card([dbc.CardBody([html.H6("Most Common Weather", className="text-center"), html.Div(id='common-weather', className='card-metric text-center')])], className='mb-2', style=card_style),
            dbc.Card([dbc.CardBody([html.H6("Peak Hour", className="text-center"), html.Div(id='peak-hour', className='card-metric text-center')])], className='mb-2', style=card_style),
            dbc.Card([dbc.CardBody([html.H6("Most Severe Zone", className="text-center"), html.Div(id='severe-zone', className='card-metric text-center')])], style=card_style)
        ], style={'width': '15%', 'padding': '10px', 'display': 'flex', 'flexDirection': 'column'}),

        html.Div([
            html.Button("🔍 Fullscreen Map", id='fullscreen-btn', n_clicks=0, className="btn btn-outline-primary", style={'marginBottom': '5px'}),
            html.Div([
                dcc.Graph(id='map-graph', config={'displayModeBar': True}, style={'height': '100%', 'width': '100%'})
            ], id='map-container', style={'position': 'relative','height': '600px', 'width': '100%'})
        ], style={'width': '65%', 'padding': '10px'}),

        html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Predictive Model Output (Coming Soon)", style={
                        'textAlign': 'center',
                        'margin': 0,
                        'fontSize': '14px'
                    })
                ], style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'height': '100%'
                })
            ], style={'height': '600px'})
        ], style={'width': '20%', 'padding': '10px'})
    ], style={'display': 'flex'}),

    # Bottom Rows: Charts
    html.Div([
        html.Div(dbc.Card(dbc.CardBody(dcc.Graph(id='time-series-graph', style=chart_style))), style={'width': '33.3%', 'padding': '5px'}),
        html.Div(dbc.Card(dbc.CardBody(dcc.Graph(id='weather-trend-graph', style=chart_style))), style={'width': '33.3%', 'padding': '5px'}),
        html.Div(dbc.Card(dbc.CardBody(dcc.Graph(id='weather-bar-graph', style=chart_style))), style={'width': '33.3%', 'padding': '5px'})
    ], style={'display': 'flex', 'padding': '10px'}),

    html.Div([
        html.Div(dbc.Card(dbc.CardBody(dcc.Graph(id='zone-bar-graph', style=chart_style))), style={'width': '33.3%', 'padding': '5px'}),
        html.Div(dbc.Card(dbc.CardBody(dcc.Graph(id='hour-bar-graph', style=chart_style))), style={'width': '33.3%', 'padding': '5px'}),
        html.Div(dbc.Card(dbc.CardBody(dcc.Graph(id='part-of-day-line', style=chart_style))), style={'width': '33.3%', 'padding': '5px'})
    ], style={'display': 'flex', 'padding': '10px'}),

    html.Div([
    html.Img(src='https://img.icons8.com/ios-filled/50/robot-2.png',
             id='chatbot-image',
             style={'width': '40px', 'cursor': 'pointer'}),
    dbc.Tooltip("Chatbot Feature Coming Soon", target="chatbot-image", placement="top")
], style={'position': 'fixed', 'bottom': '10px', 'left': '10px'}),
    dcc.Store(id='fullscreen-trigger')
])

# === CALLBACKS ===
@app.callback(
    Output('map-graph', 'figure'),
    Output('time-series-graph', 'figure'),
    Output('weather-trend-graph', 'figure'),
    Output('zone-bar-graph', 'figure'),
    Output('weather-bar-graph', 'figure'),
    Output('hour-bar-graph', 'figure'),
    Output('part-of-day-line', 'figure'),
    Output('total-accidents', 'children'),
    Output('common-weather', 'children'),
    Output('peak-hour', 'children'),
    Output('severe-zone', 'children'),
    Input('day-filter', 'value'),
    Input('month-filter', 'value'),
    Input('year-filter', 'value'),
    Input('season-filter', 'value'),
    Input('severity-filter', 'value'),
    Input('time-range-slider', 'value')
)
def update_dashboard(day, month, year, seasons, severities, time_range):
    filtered = df.copy()
    if day:
        filtered = filtered[filtered['acci_day'].isin(day)]
    if month:
        filtered = filtered[filtered['acci_month'].isin(month)]
    if year:
        filtered = filtered[filtered['year'].isin(year)]
    if seasons:
        filtered = filtered[filtered['season'].isin(seasons)]
    if severities:
        filtered = filtered[filtered['severity'].isin(severities)]
    filtered = filtered[(filtered['acci_hour'] >= time_range[0]) & (filtered['acci_hour'] <= time_range[1])]

    map_fig = px.density_mapbox(filtered, lat='acci_x', lon='acci_y', radius=10,
                                center=dict(lat=25.2, lon=55.3), zoom=9,
                                mapbox_style="open-street-map",
                                hover_data={'condition': True, 'severity': True, 'temp_c': True, 'humidity': True, 'wind_kph': True})

    time_fig = px.histogram(filtered, x='acci_hour', color='severity', nbins=24, title="Accidents per Hour")
    weather_fig = px.scatter(filtered, x='temp_c', y='humidity', color='severity',
                             title="Temperature vs Humidity", hover_data=['condition', 'wind_kph'])

    condition_counts = filtered['condition'].value_counts()
    total = condition_counts.sum()
    percentages = condition_counts / total * 100
    threshold = 1.0
    low_pct = percentages[percentages < threshold]
    high_pct = percentages[percentages >= threshold]
    bar_labels = list(high_pct.index) + (['Other'] if not low_pct.empty else [])
    bar_values = list(high_pct.values) + ([low_pct.sum()] if not low_pct.empty else [])

    # Create a proper DataFrame for weather condition chart
    weather_data = pd.DataFrame({
        'Weather Condition': bar_labels,
        'Percentage': bar_values
    })

    # Build horizontal bar chart using DataFrame columns
    weather_bar_fig = px.bar(
        weather_data,
        x='Percentage',
        y='Weather Condition',
        orientation='h',
        labels={'Percentage': 'Percentage', 'Weather Condition': 'Weather Condition'},
        title="Weather Condition Distribution (%)",
        text=weather_data['Percentage'].apply(lambda x: f"{x:.1f}%")
    )


    zone_counts = filtered.groupby(['zone', 'severity']).size().reset_index(name='count')
    zone_fig = px.bar(zone_counts, x='zone', y='count', color='severity', barmode='stack', title="Accidents per Zone")

    hour_fig = px.bar(filtered, x='day_of_week', color='severity', title="Accidents per Day of Week",
                      category_orders={'day_of_week': weekday_order})

    part_of_day_fig = px.line(filtered.groupby('part_of_day', observed=False).size().reset_index(name='count'),
                              x='part_of_day', y='count', title="Accidents by Part of Day")

    total_val = len(filtered)
    common_weather = filtered['condition'].mode()[0] if not filtered.empty and not filtered['condition'].mode().empty else 'N/A'
    peak_hr = filtered['acci_hour'].mode()[0] if not filtered.empty and not filtered['acci_hour'].mode().empty else 'N/A'
    
    if not filtered[filtered['severity'] == 'Fatal'].empty:
        severe = filtered[filtered['severity'] == 'Fatal'].groupby('zone').size()
        severe_zone = f"Zone {severe.idxmax()}" if not severe.empty else "No Fatal Accidents"
    else:
        severe_zone = "No Fatal Accidents"

    return map_fig, time_fig, weather_fig, zone_fig, weather_bar_fig, hour_fig, part_of_day_fig, total_val, common_weather, peak_hr, severe_zone

# === CLIENTSIDE FULLSCREEN MAP ===
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='fullscreenMap'),
    Output('fullscreen-trigger', 'data'),
    Input('fullscreen-btn', 'n_clicks')
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        fullscreenMap: function(n_clicks) {
            const el = document.getElementById('map-container');
            if (!el || !n_clicks) return window.dash_clientside.no_update;

            const plot = el.querySelector('div.js-plotly-plot');
            if (!plot) return window.dash_clientside.no_update;

            // Save original styles to restore later
            const originalHeight = el.style.height;
            const originalPlotHeight = plot.style.height;
            const originalPlotWidth = plot.style.width;

            function requestFullScreen(elem) {
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                } else if (elem.webkitRequestFullscreen) {
                    elem.webkitRequestFullscreen();
                } else if (elem.msRequestFullscreen) {
                    elem.msRequestFullscreen();
                }
            }

            function exitHandler() {
                if (!document.fullscreenElement) {
                    // Restore original styles
                    el.style.height = originalHeight;
                    plot.style.height = originalPlotHeight;
                    plot.style.width = originalPlotWidth;

                    document.removeEventListener("fullscreenchange", exitHandler);
                }
            }

            // Set fullscreen styles
            el.style.height = "100vh";
            plot.style.height = "100vh";
            plot.style.width = "100vw";

            // Enter fullscreen and listen for exit
            requestFullScreen(el);
            document.addEventListener("fullscreenchange", exitHandler);

            return '';
        }
    }
});
</script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
