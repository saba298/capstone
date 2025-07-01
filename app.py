# app.py

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from layout.overview_tab import get_layout as get_overview_layout, register_callbacks as register_overview_callbacks
from layout.predict_tab import get_layout as get_predict_layout, register_callbacks as register_predict_callbacks
from layout.future_forecast import get_layout as get_forecast_layout, register_callbacks as register_forecast_callbacks
from layout.risk_forecast import get_layout as get_risk_layout, register_callbacks as register_risk_callbacks

# Use a Bootswatch theme (e.g., LUX)
external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Traffic Accident Dashboard"

# Layout with vertical tab navigation
app.layout = html.Div([
    
    dbc.Container([
        dbc.Row([
            # Left sidebar with title and vertical tabs
            dbc.Col([
                # Dashboard title at top left
                html.H3("Safe Road AI", className="mb-4", 
                       style={'color': 'white', 'font-weight': 'bold', 'text-align': 'left', 'font-size': '1.8rem', 'padding': '20px 20px 0 20px'}),
                
                # Center container for tabs only
                html.Div([
                    # Vertical tabs with improved styling
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Overview", href="#", id="tab-1-link", active=True, className="mb-3 sidebar-tab", 
                                              style={'color': 'white', 'border': 'none', 'border-radius': '12px', 'text-align': 'center', 'font-weight': '500'})),
                        dbc.NavItem(dbc.NavLink("Predict", href="#", id="tab-2-link", className="mb-3 sidebar-tab",
                                              style={'color': 'white', 'border': 'none', 'border-radius': '12px', 'text-align': 'center', 'font-weight': '500'})),
                        dbc.NavItem(dbc.NavLink("Forecast", href="#", id="tab-3-link", className="mb-3 sidebar-tab",
                                              style={'color': 'white', 'border': 'none', 'border-radius': '12px', 'text-align': 'center', 'font-weight': '500'})),
                        dbc.NavItem(dbc.NavLink("Risk Analysis", href="#", id="tab-4-link", className="mb-3 sidebar-tab",
                                              style={'color': 'white', 'border': 'none', 'border-radius': '12px', 'text-align': 'center', 'font-weight': '500'})),
                    ], vertical=True, pills=True, id="nav-tabs"),
                ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'height': 'calc(100vh - 100px)', 'padding': '0 20px'}),
                
                # Hidden store to track active tab
                dcc.Store(id="active-tab", data="tab-1"),
            ], width=2, style={'background': 'transparent', 'min-height': '100vh'}),
            
            # Main content area
            dbc.Col([
                html.Div(id='tabs-content', className="p-3"),
            ], width=10),
        ], className="g-0"),
    ], fluid=True, style={'margin': '0', 'padding': '0'}),
], style={'background-image': 'linear-gradient(to right, #243949 0%, #517fa4 100%)', 'min-height': '100vh'})


# Callback to handle tab switching
@app.callback(
    [Output("tab-1-link", "active"),
     Output("tab-2-link", "active"),
     Output("tab-3-link", "active"),
     Output("tab-4-link", "active"),
     Output("active-tab", "data")],
    [Input("tab-1-link", "n_clicks"),
     Input("tab-2-link", "n_clicks"),
     Input("tab-3-link", "n_clicks"),
     Input("tab-4-link", "n_clicks")],
    prevent_initial_call=False
)
def toggle_tabs(n1, n2, n3, n4):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, False, False, "tab-1"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "tab-1-link":
        return True, False, False, False, "tab-1"
    elif button_id == "tab-2-link":
        return False, True, False, False, "tab-2"
    elif button_id == "tab-3-link":
        return False, False, True, False, "tab-3"
    elif button_id == "tab-4-link":
        return False, False, False, True, "tab-4"
    
    return True, False, False, False, "tab-1"


@app.callback(
    Output('tabs-content', 'children'),
    Input('active-tab', 'data')
)
def render_tab(active_tab):
    if active_tab == 'tab-1':
        return get_overview_layout()
    elif active_tab == 'tab-2':
        return get_predict_layout()
    elif active_tab == 'tab-3':
        return get_forecast_layout()
    elif active_tab == 'tab-4':
        return get_risk_layout()
    return html.Div("Tab not found.")


# Register callbacks for all tabs
register_overview_callbacks(app)
register_predict_callbacks(app)
register_forecast_callbacks(app)
register_risk_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)