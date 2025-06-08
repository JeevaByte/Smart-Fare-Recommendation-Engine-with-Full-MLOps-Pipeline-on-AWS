# PowerShell script to deploy the LangChain agent for user interaction

# Set environment variables
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:API_ENDPOINT = "http://localhost:8000"  # Local testing
# $env:API_ENDPOINT = "http://your-ecs-service-url"  # Production

# Install Streamlit if not already installed
pip install streamlit langchain openai

# Run the LangChain agent with Streamlit
Write-Host "Starting LangChain agent with Streamlit interface..."
streamlit run langchain_agent/fare_agent.py -- --api-endpoint $env:API_ENDPOINT

Write-Host "LangChain agent deployed successfully!"