#!/bin/bash
# Script to deploy the LangChain agent for user interaction

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export API_ENDPOINT="http://localhost:8000"  # Local testing
# export API_ENDPOINT="http://your-ecs-service-url"  # Production

# Install Streamlit if not already installed
pip install streamlit langchain openai

# Run the LangChain agent with Streamlit
echo "Starting LangChain agent with Streamlit interface..."
streamlit run langchain_agent/fare_agent.py -- --api-endpoint $API_ENDPOINT

echo "LangChain agent deployed successfully!"