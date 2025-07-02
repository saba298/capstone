
from utils.chatbot_utils import create_gemini_assistant, process_chatbot_query
import dash
from dash import callback, Input, Output, State, html, dcc, callback_context, clientside_callback
import dash_bootstrap_components as dbc
import pandas as pd
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Import Gemini chatbot utilities (same as your code)
try:
    from utils.chatbot_utils import create_gemini_assistant
    GEMINI_CHATBOT_AVAILABLE = True
    print("✅ Gemini chatbot utilities loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import Gemini chatbot utilities: {e}")
    GEMINI_CHATBOT_AVAILABLE = False
    try:
        from utils.chatbot_utils import process_chatbot_query
        ORIGINAL_CHATBOT_AVAILABLE = True
        print("✅ Original chatbot utilities loaded as fallback")
    except ImportError:
        ORIGINAL_CHATBOT_AVAILABLE = False
        print("❌ No chatbot utilities available")

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_CHATBOT_AVAILABLE:
    assistant = create_gemini_assistant(api_key=GEMINI_API_KEY)
    print(f"🤖 Gemini Assistant Status: {'Enabled' if assistant.gemini_available else 'Fallback Mode'}")

def load_data():
    # Same data loading logic as in your code...
    possible_paths = [
        'C:\\Users\\LENOVO\\Desktop\\Aiproject\\dash\\dashboard\\data\\cleaned_dataset.xlsx'
    ]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
                print(f"✅ Data loaded from: {path}")
                break
            except Exception as e:
                print(f"❌ Failed to load from {path}: {e}")
                continue
    if df is None:
        print("❌ Could not load data from any path")
        return pd.DataFrame(), False
    
    # Preprocessing same as your code (dropping columns, etc.)
    columns_to_drop = ['acci_id', 'acci_ar', 'acci_en', 'description']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    initial_rows = len(df)
    if 'acci_x' in df.columns and 'acci_y' in df.columns:
        df.dropna(subset=['acci_x', 'acci_y'], inplace=True)
        print(f"📍 Removed {initial_rows - len(df)} rows with missing coordinates")

    if 'acci_hour' not in df.columns and 'acci_time' in df.columns:
        try:
            df['acci_hour'] = pd.to_datetime(df['acci_time'], format='%H:%M:%S', errors='coerce').dt.hour
            df['acci_hour'].fillna(0, inplace=True)
            print("⏰ Created acci_hour column from acci_time")
        except Exception as e:
            print(f"❌ Error creating hour column: {e}")

    if 'acci_date' in df.columns:
        try:
            df['acci_date'] = pd.to_datetime(df['acci_date'], errors='coerce')
            df['acci_day'] = df['acci_date'].dt.day
            df['acci_month'] = df['acci_date'].dt.month
            df['year'] = df['acci_date'].dt.year
            df.dropna(subset=['acci_date'], inplace=True)
            print("📅 Processed date columns successfully")
        except Exception as e:
            print(f"❌ Error processing dates: {e}")
    print(f"✅ Final data shape: {df.shape}")
    return df, True

df, DATA_LOADED = load_data()

EXAMPLE_QUERIES = {
    'example-1': "Show me severe accidents in March and tell me what patterns you notice",
    'example-2': "Analyze rain-related accidents in zone 2 - what should drivers know?", 
    'example-3': "Compare summer vs winter accident patterns in 2022",
    'example-4': "What are the peak accident hours and why do you think that happens?",
    'example-5': "Weekend vs weekday accidents - give me insights for city planning",
    'example-6': "Weather impact on accidents - create a safety report",
    'example-7': "Which zones need more traffic safety measures?",
    'example-8': "Time-based accident trends - when should people be most careful?"
}

def create_message_bubble(sender, message, color, user=False):
    formatted_message = format_message_content(message)
    
    if user:
        bubble_class = "user-message"
    elif color == "danger":
        bubble_class = "error-message"
    elif color == "info":
        bubble_class = "info-message"
    else:
        bubble_class = "ai-message"
    
    return dbc.Alert([
        html.Div([
            html.Strong(f"{sender}: "),
            formatted_message
        ])
    ], color=color, className=f"mb-2 {bubble_class} fade-in-up")

def format_message_content(message):
    if not isinstance(message, str):
        return html.Span(str(message))
    lines = message.split('\n')
    formatted_elements = []
    for i, line in enumerate(lines):
        if not line.strip():
            formatted_elements.append(html.Br())
            continue
        if line.startswith('**') and line.endswith('**') and len(line) > 4:
            formatted_elements.append(html.Strong(line[2:-2]))
        elif line.startswith('• ') or line.startswith('- '):
            formatted_elements.append(html.Li(line[2:]))
        elif line.startswith('##'):
            formatted_elements.append(html.H6(line[2:].strip(), className="mt-2 mb-1"))
        elif line.startswith('#'):
            formatted_elements.append(html.H5(line[1:].strip(), className="mt-2 mb-1"))
        else:
            formatted_elements.append(html.Span(line))
        if i < len(lines) - 1:
            formatted_elements.append(html.Br())
    return html.Div(formatted_elements)

def process_chat_query(query, current_messages):
    if not DATA_LOADED:
        error_response = create_message_bubble("AI Assistant", 
            "❌ Sorry, I can't process queries right now because the dataset isn't loaded properly. Please check the data file and refresh the page.", 
            "danger")
        return current_messages + [error_response]
    
    user_message = create_message_bubble("👤 You", query, "primary", user=True)
    
    try:
        if GEMINI_CHATBOT_AVAILABLE:
            response = assistant.process_chatbot_query(query, df)
        elif ORIGINAL_CHATBOT_AVAILABLE:
            response = process_chatbot_query(query, df)
        else:
            response = "❌ Sorry, chatbot functionality is not available. Please check the installation of required modules."
        ai_message = create_message_bubble("🤖 AI Assistant", response, "success")
        return current_messages + [user_message, ai_message]
    except Exception as e:
        error_message = create_message_bubble("🤖 AI Assistant", 
            f"🔧 I encountered an error while processing your question: {str(e)}\n\nPlease try rephrasing your question or ask about specific aspects like severity, weather patterns, or time trends.", 
            "danger")
        return current_messages + [user_message, error_message]

def create_layout():
    if not DATA_LOADED:
        return dbc.Container([
            dbc.Alert([
                html.H4("⚠️ Data Loading Issue", className="alert-heading"),
                html.P("Unable to load the traffic accident dataset. Please check the file path and try again."),
            ], color="warning", className="status-alert")
        ], className="main-container")
    
    chatbot_status = "🤖 Gemini AI" if (GEMINI_CHATBOT_AVAILABLE and assistant.gemini_available) else (
        "🔄 Hybrid Mode" if GEMINI_CHATBOT_AVAILABLE else (
            "📊 Basic Assistant" if ORIGINAL_CHATBOT_AVAILABLE else "❌ Limited"
        )
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("SafeRoad AI Assistant", className="app-title"),
                    html.P("Powered by Gemini AI • Dubai Traffic Accident Analysis", 
                           className="app-subtitle"),
                ], className="app-header"),
                
                dbc.Alert([
                    html.Div([
                        html.Strong(f"{chatbot_status}: "),
                        html.Span("Advanced natural language traffic accident assistant" if GEMINI_CHATBOT_AVAILABLE else "Basic query processing chatbot")
                    ]),
                    html.Hr(className="my-2"),
                    html.Small([
                        html.Strong("Dataset: "),
                        f"{len(df):,} traffic accidents • ",
                        f"{df['acci_date'].min().strftime('%Y')} - {df['acci_date'].max().strftime('%Y')}"
                    ])
                ], color="info", className="status-alert mb-4"),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("💬 Intelligent Chat Assistant", className="mb-0 d-inline"),
                            dbc.ButtonGroup([
                                dbc.Button("🗑️ Clear", id="clear-chat-button", color="outline-secondary", size="sm"),
                                dbc.Button("💡 Tips", id="tips-button", color="outline-info", size="sm"),
                            ], className="float-end header-btn-group")
                        ])
                    ]),
                    dbc.CardBody([
                        html.Div(id='chat-messages-enhanced', children=[
                            dbc.Alert([
                                html.Div([
                                    html.Strong("🤖 AI Assistant: "),
                                    html.Span("Hello! 👋 I'm your intelligent traffic data assistant. "),
                                    html.Br(),
                                    html.Span("I can analyze accident patterns, provide insights, and answer complex questions about Dubai traffic safety. "),
                                    html.Br(),
                                    html.Small("Try asking detailed questions - I can understand context and provide comprehensive analysis!", 
                                             className="text-muted")
                                ])
                            ], color="info", className="info-message")
                        ], className="chat-messages-container"),
                        
                        html.Hr(),
                        
                        dbc.InputGroup([
                            dbc.Input(
                                id='chat-input-enhanced',
                                type='text',
                                placeholder='Ask anything about traffic accidents... (e.g., "Why do more accidents happen in March?")',
                                className="chat-input",
                                debounce=True
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-paper-plane me-1"), "Send"], 
                                id='chat-send-enhanced', 
                                className="chat-send-btn pulse-animation"
                            )
                        ], className="chat-input-group mb-3"),
                        
                        html.Div([
                            html.H6("💡 Try these intelligent queries:", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Severe accidents patterns", id="example-1", className="example-btn mb-2 w-100"),
                                    dbc.Button("Weather safety analysis", id="example-2", className="example-btn mb-2 w-100"),
                                    dbc.Button("Seasonal comparisons", id="example-3", className="example-btn mb-2 w-100"),
                                    dbc.Button("Peak hours insights", id="example-4", className="example-btn mb-2 w-100"),
                                ], md=6),
                                dbc.Col([
                                    dbc.Button("Weekend vs weekday trends", id="example-5", className="example-btn mb-2 w-100"),
                                    dbc.Button("Weather impact report", id="example-6", className="example-btn mb-2 w-100"),
                                    dbc.Button("Zone safety assessment", id="example-7", className="example-btn mb-2 w-100"),
                                    dbc.Button("Time-based safety tips", id="example-8", className="example-btn mb-2 w-100"),
                                ], md=6)
                            ])
                        ], className="example-queries-section")
                    ])
                ], className="chat-card mb-4"),
                
                dbc.Modal([
                    dbc.ModalHeader("💡 AI Assistant Tips"),
                    dbc.ModalBody([
                        html.H6("🎯 What makes this AI special:"),
                        html.Ul([
                            html.Li("Natural language understanding - ask questions like you would to a human expert"),
                            html.Li("Context awareness - I remember what we've discussed in this session"),
                            html.Li("Multi-factor analysis - I can combine time, weather, location, and severity data"),
                            html.Li("Actionable insights - I provide practical recommendations, not just numbers"),
                        ]),
                        html.Hr(),
                        html.H6("📝 Example questions that work great:"),
                        html.Ul([
                            html.Li("\"Why do you think accidents peak at 7 AM?\""),
                            html.Li("\"What safety measures would you recommend for Zone 3?\""),
                            html.Li("\"Compare accident patterns between summer and winter\""),
                            html.Li("\"How does weather really affect driving safety?\""),
                            html.Li("\"What can city planners learn from this data?\""),
                        ]),
                        html.Hr(),
                        html.Small("💡 Tip: The more specific your question, the more detailed insights I can provide!", 
                                 className="text-muted")
                    ]),
                    dbc.ModalFooter([
                        dbc.Button("Got it!", id="close-tips", className="ms-auto")
                    ])
                ], id="tips-modal", size="lg"),
            ], width=12)
        ])
    ], fluid=True, className="main-container fade-in-up")

# Instantiate Dash app with external stylesheets including your CSS file
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    '/assets/chatbot_styles.css'  # Your custom CSS file
])
app.title = "SafeRoad AI Chatbot"
app.layout = create_layout()

# Callback for chat messages and input clearing
@app.callback(
    [Output('chat-messages-enhanced', 'children'),
     Output('chat-input-enhanced', 'value')],
    [Input('chat-send-enhanced', 'n_clicks'),
     Input('chat-input-enhanced', 'n_submit'),
     Input('clear-chat-button', 'n_clicks')] + 
    [Input(f'example-{i}', 'n_clicks') for i in range(1, 9)],
    [State('chat-input-enhanced', 'value'),
     State('chat-messages-enhanced', 'children')],
    prevent_initial_call=True
)
def handle_chat_interaction(*args):
    send_clicks = args[0]
    input_submit = args[1] 
    clear_clicks = args[2]
    example_clicks = args[3:11]
    current_input = args[11]
    current_messages = args[12]

    ctx = callback_context
    if not ctx.triggered:
        return current_messages or [], current_input or ""
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'clear-chat-button':
        initial_message = [
            dbc.Alert([
                html.Div([
                    html.Strong("🤖 AI Assistant: "),
                    html.Span("Chat cleared! 🧹 Ready for new questions about traffic accidents. "),
                    html.Br(),
                    html.Small("What would you like to know about Dubai traffic safety?", className="text-muted")
                ])
            ], color="info", className="info-message fade-in-up")
        ]
        return initial_message, ""

    if trigger_id.startswith('example-'):
        query = EXAMPLE_QUERIES.get(trigger_id, "")
        new_messages = process_chat_query(query, current_messages or [])
        return new_messages, ""

    if trigger_id in ['chat-send-enhanced', 'chat-input-enhanced']:
        if current_input and current_input.strip():
            new_messages = process_chat_query(current_input.strip(), current_messages or [])
            return new_messages, ""

    return current_messages or [], current_input or ""

# Callback for Tips modal toggle
@app.callback(
    Output("tips-modal", "is_open"),
    [Input("tips-button", "n_clicks"), Input("close-tips", "n_clicks")],
    [State("tips-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_tips_modal(tips_clicks, close_clicks, is_open):
    if tips_clicks or close_clicks:
        return not is_open
    return is_open

# Client-side callback for auto-scroll
app.clientside_callback(
    """
    function(messages) {
        if (messages && messages.length > 0) {
            setTimeout(function() {
                try {
                    const chatContainer = document.querySelector('.chat-messages-container');
                    if (chatContainer) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                } catch (error) {
                    console.log('Auto-scroll error:', error);
                }
            }, 200);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-messages-enhanced', 'style'),
    Input('chat-messages-enhanced', 'children')
)

if __name__ == "__main__":
    app.run(debug=True, port=8051)
