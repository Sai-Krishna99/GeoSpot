import urllib.parse
from dotenv import load_dotenv
import os
import json
import asyncio
import logging
import traceback
import requests
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Content generation tools

async def generate_narrative_tool(analysis_data: str, business_type: str = "coffee shop", city: str = "Plano") -> str:
    """Generate strategic narrative using Mistral via AIML API"""
    try:
        logger.info(f"Generating narrative for {business_type} in {city}")
        
        # Get AIML API key
        aiml_api_key = os.getenv("MODEL_API_KEY")
        if not aiml_api_key:
            return json.dumps({"error": "AIML API key not found"})
        
        # System prompt for Mistral
        sys_prompt = (
            "You are a world-class retail site selection strategist with 20+ years of experience. "
            "Given scored neighborhood data, write a compelling, data-driven briefing that sounds like "
            "it comes from a top consulting firm. Include: "
            "1) Executive summary with clear recommendation "
            "2) Rank-ordered analysis of top 3 neighborhoods with specific pros/cons "
            "3) Key risk factors and mitigation strategies "
            "4) Next steps for the entrepreneur. "
            "Be confident, specific, and actionable. Use the data to tell a compelling story."
        )
        
        # Use Mistral for narrative generation
        headers = {"Authorization": f"Bearer {aiml_api_key}", "Content-Type": "application/json"}
        body = {
            "model": "mistralai/mistral-7b-instruct",  # Use Mistral via AIML
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Generate a strategic briefing for opening a {business_type} in {city}. Analysis data: {analysis_data}"}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        response = requests.post("https://api.aimlapi.com/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        
        result = response.json()
        narrative = result["choices"][0]["message"]["content"]
        
        logger.info("Strategic narrative generated successfully")
        return json.dumps({"narrative": narrative, "status": "success"})
        
    except Exception as e:
        logger.error(f"Error generating narrative: {str(e)}")
        return json.dumps({"error": str(e)})

async def generate_audio_tool(narrative_text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> str:
    """Generate audio briefing using ElevenLabs via AIML API"""
    try:
        logger.info("Generating audio briefing")
        
        # Get AIML API key
        aiml_api_key = os.getenv("MODEL_API_KEY")
        if not aiml_api_key:
            return json.dumps({"error": "AIML API key not found"})
        
        # Summarize narrative for audio (60-90 seconds)
        summary_prompt = (
            "Create a 60-90 second executive summary audio script from this strategic briefing. "
            "Focus on the top recommendation, key data points, and next steps. "
            "Write in a confident, professional tone suitable for audio narration."
        )
        
        # First, create a summary suitable for audio
        headers = {"Authorization": f"Bearer {aiml_api_key}", "Content-Type": "application/json"}
        summary_body = {
            "model": "openai/gpt-4.1-2025-04-14",
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": narrative_text}
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = requests.post("https://api.aimlapi.com/chat/completions", headers=headers, json=summary_body)
        response.raise_for_status()
        audio_script = response.json()["choices"][0]["message"]["content"]
        
        # Generate audio using ElevenLabs via AIML
        audio_body = {
            "model": "elevenlabs/eleven-multilingual-v2",
            "voice": {"voice_id": voice_id},
            "text": audio_script,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True
            }
        }
        
        audio_response = requests.post("https://api.aimlapi.com/tts", headers=headers, json=audio_body)
        audio_response.raise_for_status()
        
        # For demo purposes, return the script and status
        # In production, you'd save the audio file and return the path
        logger.info("Audio briefing generated successfully")
        return json.dumps({
            "audio_script": audio_script,
            "audio_status": "generated",
            "voice_id": voice_id,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return json.dumps({"error": str(e)})

# Content tools temporarily disabled - will be re-enabled after fixing LangChain compatibility
# def create_content_tools():
#     """Create content generation tools in LangChain format"""
#     # Content generation functionality will be implemented directly in agent prompts for now



def get_tools_description(tools):
    """Get description of available tools"""
    descriptions = []
    for tool in tools:
        if hasattr(tool, 'args'):
            # Coral tools have args attribute
            schema = json.dumps(tool.args).replace('{', '{{').replace('}', '}}')
            descriptions.append(f"Tool: {tool.name}, Schema: {schema}")
        else:
            # Content tools don't have args, use description
            descriptions.append(f"Tool: {tool.name}, Description: {tool.description}")
    return "\n".join(descriptions)

def load_config():
    """Load configuration from environment"""
    logger.info("Starting configuration loading...")
    
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME")
    logger.info(f"CORAL_ORCHESTRATION_RUNTIME: {runtime}")
    
    if not runtime:
        logger.info("Runtime not found, loading .env file...")
        load_dotenv()
        logger.info(".env file loaded successfully")
    
    logger.info("Building configuration dictionary...")
    config = {
        "runtime": runtime,
        "agent_id": os.getenv("CORAL_AGENT_ID", "geospot-content-generator"),
        "model_name": "openai/gpt-4.1-2025-04-14",
        "model_provider": os.getenv("CORAL_MODEL_PROVIDER", "aiml"),
        "api_key": os.getenv("MODEL_API_KEY"),
        "coral_sse_url": os.getenv("CORAL_SSE_URL", "http://localhost:5555/sse/v1/devmode/geospot/privkey/session1"),
        "coral_connection_url": os.getenv("CORAL_CONNECTION_URL")
    }
    
    logger.info("Configuration loaded:")
    for key, value in config.items():
        if key == "api_key":
            logger.info(f"  - {key}: {'***' if value else 'None'}")
        else:
            logger.info(f"  - {key}: {value}")
    
    logger.info("Validating required fields...")
    required_fields = ["agent_id", "model_name", "model_provider", "api_key"]
    for field in required_fields:
        if not config[field]:
            raise ValueError(f"Missing required field: {field}")
    
    logger.info("All required fields present")
    return config

async def create_agent(coral_tools):
    coral_tools_description = get_tools_description(coral_tools)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are the GeoSpot Content Generator Agent interacting with Coral Server tools. Your task is to create strategic business narratives and audio briefings from location analysis data.

Follow these steps in order:
1. Call wait_for_mentions from coral tools (timeoutMs: 30000) to receive mentions from other agents.
2. When you receive a mention, keep the thread ID and the sender ID.
3. Take 2 seconds to think about the content (instruction) and extract:
   - Location analysis data from the data analyzer
   - Business type and city information
   - Neighborhood scores and recommendations
4. Create a comprehensive strategic narrative that includes:
   - Executive summary with clear recommendation
   - Top 3 neighborhood analysis with pros/cons
   - Key risk factors and mitigation strategies
   - Next steps for the entrepreneur
5. Generate content using your knowledge and the provided analysis data. Create compelling, data-driven briefings that sound professional and actionable.
6. Format the response as a strategic business briefing with:
   - EXECUTIVE SUMMARY
   - LOCATION ANALYSIS
   - RECOMMENDATIONS
   - NEXT STEPS
7. Use send_message from coral tools to send the strategic narrative to the sender ID in the same thread ID.
8. If any error occurs, use send_message to send error details to the sender.
9. Always respond back to the sender agent with either the narrative or error information.
10. Wait for 2 seconds and repeat from step 1.

You have access to create professional business narratives and can synthesize location data into strategic recommendations. Focus on being data-driven, specific, and actionable in your content generation.

These are the coral tools: {coral_tools_description}"""
        ),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=os.getenv("MODEL_TEMPERATURE", "0.7"),
        max_tokens=os.getenv("MODEL_MAX_TOKENS", "8000"),
        base_url=os.getenv("MODEL_BASE_URL", "https://api.aimlapi.com/v1")
    )
    
    agent = create_tool_calling_agent(model, coral_tools, prompt)
    return AgentExecutor(agent=agent, tools=coral_tools, verbose=True, handle_parsing_errors=True)

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "GeoSpot content generator that creates strategic narratives and audio briefings from location analysis data"
    }

    query_string = urllib.parse.urlencode(coral_params)
    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    print(f"Connecting to Coral Server: {CORAL_SERVER_URL}")

    timeout = float(os.getenv("TIMEOUT_MS", "300"))
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_SERVER_URL,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )

    print("Multi Server Connection Established")

    coral_tools = await client.get_tools(server_name="coral")
    print(f"Coral tools count: {len(coral_tools)}")

    agent_executor = await create_agent(coral_tools)

    while True:
        try:
            print("Starting new agent invocation")
            await agent_executor.ainvoke({"agent_scratchpad": []})
            print("Completed agent invocation, restarting loop")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in agent loop: {str(e)}")
            print(traceback.format_exc())
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())