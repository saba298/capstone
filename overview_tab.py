# layout/overview_tab.py

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# === 1. Load Excel file ===
df = pd.read_excel(
    r'C:\\Users\\LENOVO\\Desktop\\Aiproject\\dash\\dashboard\\data\\cleaned_dataset.xlsx',
    parse_dates=['acci_date']
)

# === 2. Drop rows with nulls in key weather columns ===
weather_columns = ['temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb', 'condition']
df = df.dropna(subset=weather_columns)

# Optional: Reset index after cleaning
df = df.reset_index(drop=True)

# === 4. Fixed colors for severity ===
severity_color_map = {
    'simple': '#28a745',   # green
    'minor': '#ffc107',    # amber
    'severe': '#dc3545',   # red
}

# === 5. Layout for Tab 1 ===
def get_layout():
    return dbc.Container([
        # Compact header with title and instructions side by side
        dbc.Row([
            dbc.Col([
                html.H2("Traffic Accident Dashboard", className="mb-2", style={'color': '#2c3e50'}),
                html.P("Interactive exploration of traffic accident patterns and weather correlations", 
                      className="text-muted mb-0")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Quick Guide:", className="card-title mb-2"),
                        html.P("Use filters → View interactive charts → Analyze patterns", 
                              className="card-text small mb-0")
                    ], className="py-2")
                ], className="h-100", style={'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'})
            ], width=4)
        ], className="mb-3"),
        
        # Compact filter controls in a single row
        dbc.Card([
            dbc.CardHeader([
                html.H6("Filters", className="mb-0", style={'color': '#495057'})
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range", className="small fw-bold mb-1"),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            min_date_allowed=df['acci_date'].min(),
                            max_date_allowed=df['acci_date'].max(),
                            start_date=df['acci_date'].min(),
                            end_date=df['acci_date'].max(),
                            style={'fontSize': '12px'}
                        ),
                    ], width=12, xl=2, className="mb-2"),
                    
                    dbc.Col([
                        html.Label("Severity", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id='severity-filter',
                            options=[{'label': s.title(), 'value': s} for s in df['severity'].dropna().unique()],
                            value=[df['severity'].dropna().unique()[0]],
                            multi=True,
                            placeholder="Select severity",
                            style={'fontSize': '12px'}
                        ),
                    ], width=12, xl=2, className="mb-2"),
                    
                    dbc.Col([
                        html.Label("Weather", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id='condition-filter',
                            options=[{'label': c.title(), 'value': c} for c in df['condition'].dropna().unique()],
                            value=[df['condition'].dropna().unique()[0]],
                            multi=True,
                            placeholder="Select weather",
                            style={'fontSize': '12px'}
                        ),
                    ], width=12, xl=2, className="mb-2"),
                    
                    dbc.Col([
                        html.Label("Time Period", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id='partofday-filter',
                            options=[{'label': p.title(), 'value': p} for p in df['part_of_day'].dropna().unique()],
                            value=[],
                            multi=True,
                            placeholder="Select time",
                            style={'fontSize': '12px'}
                        ),
                    ], width=12, xl=2, className="mb-2"),
                    
                    dbc.Col([
                        html.Label("Season", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id='season-filter',
                            options=[{'label': s.title(), 'value': s} for s in df['season'].dropna().unique()],
                            value=[],
                            multi=True,
                            placeholder="Select season",
                            style={'fontSize': '12px'}
                        ),
                    ], width=12, xl=2, className="mb-2"),
                ], className="g-2")
            ], className="py-2")
        ], className="mb-3"),
        
        # Main content in a 2x3 grid layout
        dbc.Row([
            # Left column - Map and trends
            dbc.Col([
                # Map
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Accident Locations", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='map-accident-locations', style={'height': '350px'}),
                    ], className="p-2")
                ], className="mb-3"),
                
                # Time trends
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Temporal Patterns", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='line-accident-trends', style={'height': '250px'}),
                            ], width=7),
                            dbc.Col([
                                dcc.Graph(id='bar-accidents-by-hour', style={'height': '250px'}),
                            ], width=5),
                        ], className="g-1")
                    ], className="p-2")
                ], className="mb-3")
            ], width=12, lg=6),
            
            # Right column - Distributions and weather
            dbc.Col([
                # Distribution charts
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Accident Distributions", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='pie-severity', style={'height': '200px'}),
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='pie-partofday', style={'height': '200px'}),
                            ], width=6),
                        ], className="g-1 mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='bar-accidents-by-season', style={'height': '180px'}),
                            ], width=12),
                        ])
                    ], className="p-2")
                ], className="mb-3"),
                
                # Weather analysis
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Weather Analysis", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='box-weather-by-severity', style={'height': '220px'}),
                    ], className="p-2")
                ], className="mb-3")
            ], width=12, lg=6),
        ], className="mb-3"),
        
        # Bottom row - Advanced analytics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Weather Correlations", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='heatmap-weather-corr', style={'height': '300px'}),
                    ], className="p-2")
                ])
            ], width=12, lg=6, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Weather Variable Relationships", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='scattermatrix-weather', style={'height': '300px'}),
                    ], className="p-2")
                ])
            ], width=12, lg=6, className="mb-3"),
        ]),
        
    ], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})

# === 6. Register Callbacks ===
def register_callbacks(app):
    @app.callback(
        [Output('map-accident-locations', 'figure'),
         Output('line-accident-trends', 'figure'),
         Output('bar-accidents-by-hour', 'figure'),
         Output('pie-severity', 'figure'),
         Output('pie-partofday', 'figure'),
         Output('bar-accidents-by-season', 'figure'),
         Output('box-weather-by-severity', 'figure'),
         Output('heatmap-weather-corr', 'figure'),
         Output('scattermatrix-weather', 'figure')],
        [Input('date-range-picker', 'start_date'),
         Input('date-range-picker', 'end_date'),
         Input('severity-filter', 'value'),
         Input('condition-filter', 'value'),
         Input('partofday-filter', 'value'),
         Input('season-filter', 'value')]
    )
    def update_dashboard(start_date, end_date, severity_vals, condition_vals, partofday_vals, season_vals):
        # === Filter Data ===
        filtered_df = df[
            (df['acci_date'] >= start_date) &
            (df['acci_date'] <= end_date)
        ]

        if severity_vals:
            filtered_df = filtered_df[filtered_df['severity'].isin(severity_vals)]

        if condition_vals:
            filtered_df = filtered_df[filtered_df['condition'].isin(condition_vals)]

        if partofday_vals:
            filtered_df = filtered_df[filtered_df['part_of_day'].isin(partofday_vals)]

        if season_vals:
            filtered_df = filtered_df[filtered_df['season'].isin(season_vals)]

        if filtered_df.empty:
            fig_empty = px.line(title="No data available for selected filters")
            fig_empty.update_layout(
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12)
            )
            return [fig_empty] * 9

        # Common layout styling
        layout_style = dict(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=10),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        
        # Default legend style for most charts
        default_legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        )

        # === Map ===
        map_fig = px.scatter_mapbox(
            filtered_df, lat='acci_x', lon='acci_y',
            color='severity',
            color_discrete_map=severity_color_map,
            hover_data=['acci_date', 'condition', 'severity'],
            zoom=10,
            title='Accident Locations by Severity'
        )
        map_fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(size=10),
            title_font_size=12
        )

        # === Line Chart: Accidents by Date ===
        trend_df = filtered_df.groupby('acci_date').size().reset_index(name='count')
        line_fig = px.line(
            trend_df, x='acci_date', y='count', 
            title='Daily Accident Trends',
            color_discrete_sequence=['#007bff']
        )
        line_fig.update_layout(**layout_style, legend=default_legend, title_font_size=12)
        line_fig.update_traces(line=dict(width=2))

        # === Bar Chart: Accidents by Hour ===
        hourly_data = filtered_df.groupby('acci_hour').size().reset_index(name='count')
        bar_fig = px.bar(
            hourly_data, x='acci_hour', y='count', 
            title='Accidents by Hour',
            color_discrete_sequence=['#28a745']
        )
        bar_fig.update_layout(**layout_style, legend=default_legend, title_font_size=12)
        bar_fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)

        # === Pie Chart: Severity Distribution ===
        pie_severity = px.pie(
            filtered_df, names='severity', 
            title='Severity Distribution',
            color_discrete_map=severity_color_map
        )
        # Custom legend for pie chart
        pie_severity.update_layout(
            **layout_style,
            title_font_size=12,
            legend=dict(
                orientation="v", 
                yanchor="middle", 
                y=0.5, 
                xanchor="left", 
                x=1.05, 
                font=dict(size=9)
            )
        )
        pie_severity.update_traces(textfont_size=9, textposition="inside")

        # === Pie Chart: Part of Day ===
        pie_partofday = px.pie(
            filtered_df, names='part_of_day', 
            title='Time Period Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        # Custom legend for pie chart
        pie_partofday.update_layout(
            **layout_style,
            title_font_size=12,
            legend=dict(
                orientation="v", 
                yanchor="middle", 
                y=0.5, 
                xanchor="left", 
                x=1.05, 
                font=dict(size=9)
            )
        )
        pie_partofday.update_traces(textfont_size=9, textposition="inside")

        # === Bar Chart: Accidents by Season ===
        season_data = filtered_df.groupby('season').size().reset_index(name='count')
        bar_season = px.bar(
            season_data, x='season', y='count',
            title='Seasonal Distribution',
            color_discrete_sequence=['#ff6b6b']
        )
        bar_season.update_layout(**layout_style, legend=default_legend, title_font_size=12)

        # === Boxplot: Weather variables by Severity ===
        box_weather = px.box(
            filtered_df.melt(
                id_vars=['severity'],
                value_vars=['temp_c', 'humidity', 'wind_kph'],
                var_name='weather_metric',
                value_name='value'
            ),
            x='weather_metric', y='value', color='severity',
            title='Weather Conditions by Severity',
            color_discrete_map=severity_color_map
        )
        box_weather.update_layout(**layout_style, legend=default_legend, title_font_size=12)
        box_weather.update_xaxes(tickangle=45)

        # === Correlation heatmap ===
        weather_vars = ['temp_c', 'humidity', 'wind_kph', 'precip_mm', 'cloud', 'pressure_mb']
        severity_map_num = {'simple': 1, 'minor': 2, 'severe': 3}
        corr_df = filtered_df[weather_vars + ['severity']].copy()
        corr_df['severity'] = corr_df['severity'].map(severity_map_num)
        corr_matrix = corr_df.corr()

        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Weather-Severity Correlations',
            aspect="auto"
        )
        heatmap_fig.update_layout(**layout_style, legend=default_legend, title_font_size=12)
        heatmap_fig.update_coloraxes(colorbar_title_font_size=10)

        # === Scatterplot matrix ===
        scattermatrix_fig = px.scatter_matrix(
            filtered_df.sample(min(500, len(filtered_df))),  # Sample for performance
            dimensions=weather_vars[:4],  # Limit dimensions for better visibility
            color='severity',
            color_discrete_map=severity_color_map,
            title='Weather Variables by Severity'
        )
        scattermatrix_fig.update_layout(
            **layout_style,
            legend=default_legend,
            title_font_size=12,
            dragmode='select'
        )
        scattermatrix_fig.update_traces(diagonal_visible=False, marker=dict(size=3))

        return (map_fig, line_fig, bar_fig, pie_severity, pie_partofday, 
                bar_season, box_weather, heatmap_fig, scattermatrix_fig)