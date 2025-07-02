import pandas as pd
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
import google.generativeai as genai
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiTrafficAssistant:
    """Enhanced Traffic Assistant with Gemini AI integration"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        self.gemini_available = False
        
        # Initialize Gemini if API key is provided
        if api_key:
            try:
                genai.configure(api_key=api_key)
                
                # Try different model names - Google has updated their model names
                model_names = [
                    'gemini-1.5-flash',     # Latest recommended model
                    'gemini-1.5-pro',       # Pro version
                    'gemini-pro',           # Legacy name
                    'models/gemini-1.5-flash',  # With models/ prefix
                    'models/gemini-1.5-pro'     # With models/ prefix
                ]
                
                for model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        # Test the model with a simple prompt
                        test_response = self.model.generate_content("Hello")
                        self.gemini_available = True
                        logger.info(f"✅ Gemini AI initialized successfully with model: {model_name}")
                        break
                    except Exception as model_error:
                        logger.warning(f"⚠️ Failed to initialize model {model_name}: {model_error}")
                        continue
                
                if not self.gemini_available:
                    logger.error("❌ Failed to initialize any Gemini model")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini AI: {e}")
                self.gemini_available = False
        else:
            logger.warning("⚠️ No Gemini API key provided, using rule-based responses")
    
    def get_data_context(self, df):
        """Generate data context for Gemini"""
        if df.empty:
            return "No data available."
        
        context = f"""
        Dubai Traffic Accident Dataset Context:
        - Total records: {len(df):,}
        - Date range: {df['acci_date'].min().strftime('%Y-%m-%d')} to {df['acci_date'].max().strftime('%Y-%m-%d')}
        - Available columns: {', '.join(df.columns.tolist())}
        
        Key Statistics:
        """
        
        # Add severity info if available
        if 'severity' in df.columns:
            severity_counts = df['severity'].value_counts()
            context += f"- Severity levels: {', '.join([f'{k}: {v}' for k, v in severity_counts.items()])}\n"
        
        # Add weather info if available
        if 'condition' in df.columns:
            top_conditions = df['condition'].value_counts().head(3)
            context += f"- Top weather conditions: {', '.join([f'{k}: {v}' for k, v in top_conditions.items()])}\n"
        
        # Add zone info if available
        if 'zone' in df.columns:
            zone_count = df['zone'].nunique()
            context += f"- Number of zones: {zone_count}\n"
        
        return context
    
    def analyze_data_for_query(self, query, df):
        """Analyze data based on the query and return structured results"""
        try:
            # Extract filters from query
            filters = self.extract_smart_filters(query, df)
            
            # Apply filters
            filtered_df = self.apply_filters(df, filters)
            
            if filtered_df.empty:
                return {
                    'status': 'no_data',
                    'message': 'No data found matching your criteria.',
                    'filters_applied': filters
                }
            
            # Generate statistics
            stats = self.generate_statistics(filtered_df, query)
            
            return {
                'status': 'success',
                'data_count': len(filtered_df),
                'total_data': len(df),
                'filters_applied': filters,
                'statistics': stats,
                'insights': self.generate_insights(filtered_df, query)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {
                'status': 'error',
                'message': f"Error analyzing data: {str(e)}"
            }
    
    def extract_smart_filters(self, query, df):
        """Enhanced filter extraction with better NLP understanding"""
        filters = {}
        query_lower = query.lower()
        
        # Time-based filters
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in query_lower:
                filters['month'] = month_num
                break
        
        # Year extraction
        years = re.findall(r'20\d{2}', query)
        if years:
            filters['year'] = int(years[0])
        
        # Zone extraction - more flexible
        zone_patterns = [r'zone\s*(\d+)', r'area\s*(\d+)', r'sector\s*(\d+)']
        for pattern in zone_patterns:
            zone_match = re.search(pattern, query_lower)
            if zone_match:
                filters['zone'] = int(zone_match.group(1))
                break
        
        # Severity filters - more comprehensive
        if any(word in query_lower for word in ['severe', 'serious', 'fatal', 'major', 'critical']):
            filters['severity_type'] = 'severe'
        elif any(word in query_lower for word in ['minor', 'light', 'small', 'slight']):
            filters['severity_type'] = 'minor'
        
        # Weather conditions - expanded
        weather_keywords = {
            'rain': ['rain', 'rainy', 'raining', 'precipitation'],
            'clear': ['clear', 'sunny', 'sun', 'bright'],
            'cloud': ['cloud', 'cloudy', 'overcast'],
            'fog': ['fog', 'foggy', 'mist', 'misty'],
            'dust': ['dust', 'dusty', 'sandstorm']
        }
        
        for condition, keywords in weather_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                filters['weather'] = condition
                break
        
        # Time of day filters
        if any(word in query_lower for word in ['morning', 'am']):
            filters['time_period'] = 'morning'
        elif any(word in query_lower for word in ['afternoon', 'noon']):
            filters['time_period'] = 'afternoon'
        elif any(word in query_lower for word in ['evening', 'night', 'pm']):
            filters['time_period'] = 'evening'
        elif any(word in query_lower for word in ['rush hour', 'peak']):
            filters['time_period'] = 'rush'
        
        # Day type filters
        if any(word in query_lower for word in ['weekend', 'saturday', 'sunday']):
            filters['day_type'] = 'weekend'
        elif any(word in query_lower for word in ['weekday', 'workday']):
            filters['day_type'] = 'weekday'
        
        return filters
    
    def apply_filters(self, df, filters):
        """Apply extracted filters to dataframe"""
        filtered_df = df.copy()
        
        # Apply each filter
        if 'month' in filters and 'acci_month' in df.columns:
            filtered_df = filtered_df[filtered_df['acci_month'] == filters['month']]
        
        if 'year' in filters and 'year' in df.columns:
            filtered_df = filtered_df[filtered_df['year'] == filters['year']]
        
        if 'zone' in filters and 'zone' in df.columns:
            filtered_df = filtered_df[filtered_df['zone'] == filters['zone']]
        
        if 'severity_type' in filters and 'severity' in df.columns:
            if filters['severity_type'] == 'severe':
                filtered_df = filtered_df[filtered_df['severity'].str.contains(
                    'severe|serious|fatal|major|critical', case=False, na=False)]
            elif filters['severity_type'] == 'minor':
                filtered_df = filtered_df[filtered_df['severity'].str.contains(
                    'minor|light|slight|small', case=False, na=False)]
        
        if 'weather' in filters and 'condition' in df.columns:
            weather_condition = filters['weather']
            filtered_df = filtered_df[filtered_df['condition'].str.contains(
                weather_condition, case=False, na=False)]
        
        if 'time_period' in filters and 'acci_hour' in df.columns:
            time_period = filters['time_period']
            if time_period == 'morning':
                filtered_df = filtered_df[(filtered_df['acci_hour'] >= 6) & (filtered_df['acci_hour'] < 12)]
            elif time_period == 'afternoon':
                filtered_df = filtered_df[(filtered_df['acci_hour'] >= 12) & (filtered_df['acci_hour'] < 18)]
            elif time_period == 'evening':
                filtered_df = filtered_df[(filtered_df['acci_hour'] >= 18) | (filtered_df['acci_hour'] < 6)]
            elif time_period == 'rush':
                rush_hours = ((filtered_df['acci_hour'] >= 7) & (filtered_df['acci_hour'] <= 9)) | \
                           ((filtered_df['acci_hour'] >= 17) & (filtered_df['acci_hour'] <= 19))
                filtered_df = filtered_df[rush_hours]
        
        if 'day_type' in filters and 'acci_date' in df.columns:
            filtered_df['day_of_week'] = pd.to_datetime(filtered_df['acci_date']).dt.dayofweek
            if filters['day_type'] == 'weekend':
                filtered_df = filtered_df[filtered_df['day_of_week'] >= 5]
            elif filters['day_type'] == 'weekday':
                filtered_df = filtered_df[filtered_df['day_of_week'] < 5]
        
        return filtered_df
    
    def generate_statistics(self, df, query):
        """Generate relevant statistics based on the data and query"""
        stats = {}
        
        try:
            # Basic stats
            stats['total_accidents'] = len(df)
            
            # Severity breakdown
            if 'severity' in df.columns:
                stats['severity'] = df['severity'].value_counts().to_dict()
            
            # Weather conditions
            if 'condition' in df.columns:
                stats['weather'] = df['condition'].value_counts().head(5).to_dict()
            
            # Time patterns
            if 'acci_hour' in df.columns:
                stats['peak_hours'] = df['acci_hour'].value_counts().head(3).to_dict()
            
            # Monthly patterns
            if 'acci_month' in df.columns:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_counts = df['acci_month'].value_counts().sort_index()
                stats['monthly'] = {month_names[month-1]: count 
                                  for month, count in monthly_counts.items()}
            
            # Zone patterns
            if 'zone' in df.columns:
                stats['top_zones'] = df['zone'].value_counts().head(5).to_dict()
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def generate_insights(self, df, query):
        """Generate insights from the data"""
        insights = []
        
        try:
            # Peak time insight
            if 'acci_hour' in df.columns and len(df) > 0:
                peak_hour = df['acci_hour'].mode()[0]
                peak_count = len(df[df['acci_hour'] == peak_hour])
                insights.append(f"Peak accident time is {peak_hour}:00 with {peak_count} incidents")
            
            # Weather insight
            if 'condition' in df.columns and len(df) > 0:
                top_weather = df['condition'].mode()[0]
                weather_count = len(df[df['condition'] == top_weather])
                weather_pct = (weather_count / len(df)) * 100
                insights.append(f"Most accidents ({weather_pct:.1f}%) occur during {top_weather} conditions")
            
            # Severity insight
            if 'severity' in df.columns and len(df) > 0:
                severity_counts = df['severity'].value_counts()
                total = len(df)
                for severity, count in severity_counts.head(2).items():
                    pct = (count / total) * 100
                    insights.append(f"{severity} accidents: {count} cases ({pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def process_with_gemini(self, query, analysis_result, df):
        """Process query using Gemini AI for natural, conversational responses"""
        try:
            # Prepare context for Gemini
            data_context = self.get_data_context(df)
            
            # Create a comprehensive prompt
            prompt = f"""
            You are a friendly AI assistant helping users analyze Dubai traffic accident data.
            
            Data Context:
            {data_context}
            
            User Query: "{query}"
            
            Analysis Results:
            {json.dumps(analysis_result, indent=2, default=str)}
            
            Please provide a conversational, helpful response that:
            1. Directly answers the user's question
            2. Highlights key findings from the analysis
            3. Uses emojis appropriately to make it friendly
            4. Provides actionable insights when possible
            5. Suggests follow-up questions if relevant
            6. Keeps the tone professional but conversational
            
            Format your response in a clear, structured way with proper line breaks and bullet points where appropriate.
            """
            
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error with Gemini processing: {e}")
            return self.fallback_response(query, analysis_result)
    
    def fallback_response(self, query, analysis_result):
        """Fallback response when Gemini is not available"""
        if analysis_result['status'] == 'no_data':
            return f"🤔 I couldn't find any accidents matching your criteria for '{query}'. Try adjusting your search parameters or ask about different aspects of the data."
        
        if analysis_result['status'] == 'error':
            return f"⚠️ I encountered an issue: {analysis_result['message']}. Please try rephrasing your question."
        
        # Build response from analysis
        response = f"📊 **Analysis Results for: '{query}'**\n\n"
        response += f"Found **{analysis_result['data_count']:,}** accidents out of {analysis_result['total_data']:,} total records.\n\n"
        
        # Add filters applied
        if analysis_result['filters_applied']:
            response += "**🔍 Filters Applied:**\n"
            for key, value in analysis_result['filters_applied'].items():
                response += f"• {key.replace('_', ' ').title()}: {value}\n"
            response += "\n"
        
        # Add key statistics
        stats = analysis_result.get('statistics', {})
        if 'severity' in stats:
            response += "**🚨 Severity Breakdown:**\n"
            for severity, count in stats['severity'].items():
                pct = (count / analysis_result['data_count']) * 100
                response += f"• {severity}: {count} ({pct:.1f}%)\n"
            response += "\n"
        
        # Add insights
        insights = analysis_result.get('insights', [])
        if insights:
            response += "**💡 Key Insights:**\n"
            for insight in insights:
                response += f"• {insight}\n"
        
        return response
    
    def process_chatbot_query(self, query, df):
        """Main method to process chatbot queries"""
        try:
            if df.empty:
                return "❌ No data available to process your query. Please check if the dataset is loaded correctly."
            
            # Analyze the data based on the query
            analysis_result = self.analyze_data_for_query(query, df)
            
            # Generate response
            if self.gemini_available:
                response = self.process_with_gemini(query, analysis_result, df)
            else:
                response = self.fallback_response(query, analysis_result)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return f"🔧 I encountered an error while processing your query. Please try rephrasing your question or ask about specific aspects like severity, weather, time patterns, or locations.\n\nError details: {str(e)}"

# Initialize the assistant (you'll need to provide your Gemini API key)
# Get your API key from: https://makersuite.google.com/app/apikey
def create_gemini_assistant(api_key=None):
    """Factory function to create the assistant"""
    return GeminiTrafficAssistant(api_key=api_key)

# Legacy function for backward compatibility
def process_chatbot_query(query, df, api_key=None):
    """Legacy wrapper function for existing code compatibility"""
    assistant = create_gemini_assistant(api_key)
    return assistant.process_chatbot_query(query, df)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("🤖 Gemini Traffic Assistant Initialized")
    
    # Test queries
    test_queries = [
        "Show me severe accidents in March",
        "What about rainy weather accidents?",
        "Peak accident hours analysis",
        "Weekend vs weekday patterns",
        "Tell me about summer accidents in 2022"
    ]
    
    # You would load your actual dataframe here
    # df = pd.read_excel('your_data.xlsx')
    
    print("\n📝 Example queries that can be processed:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
