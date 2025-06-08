#!/usr/bin/env python
"""
LangChain agent for natural language interaction with the fare recommendation model.
"""

import os
import sys
import json
import requests
import argparse
from datetime import datetime, timedelta
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API endpoint from environment or use default
API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")

# UK train operators
TRAIN_OPERATORS = [
    'LNER', 'GWR', 'Avanti West Coast', 'CrossCountry', 
    'TransPennine Express', 'ScotRail', 'Northern', 
    'South Western Railway', 'Southeastern', 'East Midlands Railway'
]

# Major UK stations
STATIONS = [
    'London Kings Cross', 'London Euston', 'London Paddington', 'London Waterloo',
    'Manchester Piccadilly', 'Birmingham New Street', 'Edinburgh Waverley',
    'Glasgow Central', 'Leeds', 'Liverpool Lime Street', 'Bristol Temple Meads',
    'Cardiff Central', 'Newcastle', 'Sheffield', 'York', 'Nottingham',
    'Reading', 'Oxford', 'Cambridge', 'Brighton', 'Southampton Central'
]

# User types
USER_TYPES = ['standard', 'business', 'student', 'senior', 'family']

class FarePredictionTool(BaseTool):
    name = "fare_prediction"
    description = """
    Use this tool to get fare predictions for train journeys.
    The input should be a JSON object with the following fields:
    - origin_station: The origin station name
    - destination_station: The destination station name
    - booking_days_ahead: Days ahead of travel for booking
    - travel_time_minutes: Travel time in minutes
    - time_of_day: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - train_operator: Train operator name
    - travel_class: Class of travel (standard or first)
    - user_type: User type (standard, business, student, senior, family)
    - is_peak: Is peak time (0 or 1)
    - is_weekend: Is weekend (0 or 1)
    - is_holiday: Is holiday (0 or 1)
    - distance_miles: Distance in miles
    """
    
    def _run(self, query):
        """Run the fare prediction tool"""
        try:
            # Parse input JSON
            params = json.loads(query)
            
            # Make API request
            response = requests.post(
                f"{API_ENDPOINT}/predict",
                json=params
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                return f"Predicted fare: £{result['predicted_fare']:.2f} (Confidence: {result['confidence']:.2f})"
            else:
                return f"Error: API returned status code {response.status_code}: {response.text}"
        
        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except requests.RequestException as e:
            return f"Error: Failed to connect to API: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _arun(self, query):
        """Async implementation of the fare prediction tool"""
        # This could be implemented with aiohttp for async requests
        raise NotImplementedError("Async version not implemented")

class StationInfoTool(BaseTool):
    name = "station_info"
    description = """
    Use this tool to get information about train stations.
    The input should be a station name or 'list' to get all available stations.
    """
    
    def _run(self, query):
        """Run the station info tool"""
        if query.lower() == 'list':
            return f"Available stations: {', '.join(STATIONS)}"
        
        # Check if station exists
        for station in STATIONS:
            if query.lower() in station.lower():
                return f"Station found: {station}"
        
        return f"Station not found: {query}. Available stations: {', '.join(STATIONS)}"
    
    def _arun(self, query):
        """Async implementation of the station info tool"""
        raise NotImplementedError("Async version not implemented")

class TrainOperatorTool(BaseTool):
    name = "train_operator_info"
    description = """
    Use this tool to get information about train operators.
    The input should be an operator name or 'list' to get all available operators.
    """
    
    def _run(self, query):
        """Run the train operator tool"""
        if query.lower() == 'list':
            return f"Available train operators: {', '.join(TRAIN_OPERATORS)}"
        
        # Check if operator exists
        for operator in TRAIN_OPERATORS:
            if query.lower() in operator.lower():
                return f"Train operator found: {operator}"
        
        return f"Train operator not found: {query}. Available operators: {', '.join(TRAIN_OPERATORS)}"
    
    def _arun(self, query):
        """Async implementation of the train operator tool"""
        raise NotImplementedError("Async version not implemented")

def parse_natural_language_query(query):
    """
    Parse a natural language query into structured parameters for the fare prediction API.
    This is a placeholder - in a real implementation, this would use the LLM to extract parameters.
    """
    # This is a simplified example - in reality, you would use the LLM to extract these parameters
    params = {
        "origin_station": "London Kings Cross",
        "destination_station": "Edinburgh Waverley",
        "booking_days_ahead": 7,
        "travel_time_minutes": 240,
        "time_of_day": 10,
        "day_of_week": 1,
        "train_operator": "LNER",
        "travel_class": "standard",
        "user_type": "standard",
        "is_peak": 0,
        "is_weekend": 0,
        "is_holiday": 0,
        "distance_miles": 400
    }
    
    return params

def create_agent():
    """Create a LangChain agent with the fare prediction tools"""
    # Initialize the language model
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize tools
    tools = [
        FarePredictionTool(),
        StationInfoTool(),
        TrainOperatorTool()
    ]
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # System message
    system_message = SystemMessage(
        content="""You are a helpful train fare assistant. You help users find the best fares for their train journeys in the UK.
        When users ask about fares, extract the relevant information from their query and use the fare_prediction tool to get a prediction.
        If information is missing, ask follow-up questions to get the necessary details.
        Always be polite, helpful, and concise in your responses."""
    )
    
    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            "system_message": system_message,
        }
    )
    
    return agent

def run_cli():
    """Run the agent in CLI mode"""
    agent = create_agent()
    
    print("Welcome to the Train Fare Assistant!")
    print("Ask me about train fares or type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        response = agent.run(query)
        print(f"\nAssistant: {response}")

def run_streamlit():
    """Run the agent in Streamlit mode"""
    st.title("Train Fare Assistant")
    st.write("Ask me about train fares in the UK!")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize agent
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()
    
    # Get user input
    if prompt := st.chat_input("How can I help you find train fares today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    parser = argparse.ArgumentParser(description='Train Fare Assistant')
    parser.add_argument('--mode', type=str, choices=['cli', 'streamlit'], default='streamlit',
                        help='Mode to run the assistant (cli or streamlit)')
    parser.add_argument('--api-endpoint', type=str, help='API endpoint for fare predictions')
    
    args = parser.parse_args()
    
    # Set API endpoint if provided
    if args.api_endpoint:
        global API_ENDPOINT
        API_ENDPOINT = args.api_endpoint
    
    # Run in selected mode
    if args.mode == 'cli':
        run_cli()
    else:
        run_streamlit()

if __name__ == "__main__":
    main()