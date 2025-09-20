import urllib.parse
from dotenv import load_dotenv
import os
import json
import asyncio
import logging
from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate  
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Constants
MAX_CHAT_HISTORY = 3
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    logger.info("Starting configuration loading...")
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    logger.info(f"CORAL_ORCHESTRATION_RUNTIME: {runtime}")
    
    if runtime is None:
        logger.info("Runtime not found, loading .env file...")
        load_dotenv()
        logger.info(".env file loaded successfully")
    else:
        logger.info("Runtime environment detected, skipping .env file")
    
    logger.info("Building configuration dictionary...")
    config = {
        "runtime": os.getenv("CORAL_ORCHESTRATION_RUNTIME", None),
        "coral_connection_url": os.getenv("CORAL_CONNECTION_URL"),
        "coral_sse_url": os.getenv("CORAL_SSE_URL"),
        "agent_id": os.getenv("CORAL_AGENT_ID"),
        "model_name": os.getenv("MODEL_NAME"),
        "model_provider": os.getenv("MODEL_PROVIDER"),
        "api_key": os.getenv("MODEL_API_KEY"),
        "model_temperature": float(os.getenv("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE)),
        "model_token": int(os.getenv("MODEL_TOKEN_LIMIT", DEFAULT_MAX_TOKENS)),
        "base_url": "https://api.aimlapi.com/v1"  # AIML API base URL
    }
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  - runtime: {config['runtime']}")
    logger.info(f"  - agent_id: {config['agent_id']}")
    logger.info(f"  - model_name: {config['model_name']}")
    logger.info(f"  - model_provider: {config['model_provider']}")
    logger.info(f"  - api_key: {'***' if config['api_key'] else None}")
    
    logger.info("Validating required fields...")
    required_fields = ["model_name", "model_provider", "api_key"]
    missing = [field for field in required_fields if not config[field]]
    if missing:
        logger.error(f"Missing required fields: {missing}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    logger.info("All required fields present")
    
    return config

def get_tools_description(tools: List[Any]) -> str:
    """Generate description of available tools"""
    return "\\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for context"""
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    for entry in chat_history[-MAX_CHAT_HISTORY:]:
        # Fix the key names to match what we're actually storing
        formatted.append(f"User: {entry.get('user_input', 'N/A')}")
        formatted.append(f"Assistant: {entry.get('response', 'N/A')}")
    
    return "\n".join(formatted)

async def create_agent(coral_tools: List[Any]) -> AgentExecutor:
    """Create agent executor following reference pattern."""
    coral_tools_description = get_tools_description(coral_tools)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are the GeoSpot Orchestrator, coordinating business location analysis between specialized agents.

Your role:
- Understand user's business location analysis needs  
- Coordinate with Data Analyzer for demographics and scoring
- Coordinate with Content Generator for strategic narratives
- Present actionable business location insights

Use {{chat_history}} for context.

Steps for business location analysis:
1. Call list-agents to see available agents
2. Extract business type and location from user request
3. Create thread with data-analyzer and content-generator
4. Send analysis request to data analyzer first
5. Wait for data response using wait-for-mentions
6. Send content request to content generator with data
7. Wait for narrative response
8. Synthesize final recommendations

Available tools: {coral_tools_description}"""
        ),
        ("human", "{user_input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model="openai/gpt-4.1-2025-04-14",
        model_provider="openai", 
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=0.3,
        max_tokens=2000,
        base_url="https://api.aimlapi.com/v1"
    )

    agent = create_tool_calling_agent(model, coral_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=coral_tools, verbose=True, handle_parsing_errors=True)
    
    return executor

async def get_user_input(runtime: str, agent_tools: Dict[str, Any]) -> str:
    """Get user input using request-question tool in runtime mode"""
    logger.info(f"Starting user input retrieval. Runtime mode: {runtime is not None}")
    
    if runtime is not None:
        logger.info("Using runtime mode - invoking request-question tool")
        logger.info(f"Available agent tools: {list(agent_tools.keys())}")
        try:
            logger.info("Calling request_question tool with message prompt...")
            
            # Add timeout to prevent hanging
            user_input = await asyncio.wait_for(
                agent_tools["request-question"].ainvoke({
                    "message": "Welcome to GeoSpot! I'm your AI location intelligence assistant. What type of business are you looking to open, and in which city would you like me to analyze? (e.g., 'coffee shop in Plano, TX')"
                }),
                timeout=60.0  # 60 second timeout
            )
            logger.info(f"Successfully received input from runtime tool: {len(str(user_input))} chars")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for user input - UI may not be connected properly")
            # Fallback to default input for testing
            user_input = "coffee shop in Plano, TX"
            logger.info(f"Using fallback input for testing: {user_input}")
        except Exception as e:
            logger.error(f"Failed to invoke request_question tool: {str(e)}")
            raise
    else:
        logger.info("Using interactive mode - prompting user directly")
        user_input = input("Welcome to GeoSpot! What type of business are you looking to open? ").strip()
        logger.info(f"Raw user input received: '{user_input}'")
        
        if not user_input:
            logger.info("Empty input detected, using default message")
            user_input = "No input provided"
    
    logger.info(f"Final processed user input: {user_input}")
    return user_input

async def send_response(runtime: str, agent_tools: Dict[str, Any], response: str) -> None:
    """Send response using answer-question tool in runtime mode"""
    logger.info(f"Starting response sending. Runtime mode: {runtime is not None}")
    logger.info(f"Response length: {len(response)} characters")
    logger.info(f"Response preview: {response[:200]}...")
    
    if runtime is not None:
        logger.info("Using runtime mode - invoking answer-question tool")
        logger.info(f"Available agent tools: {list(agent_tools.keys())}")
        try:
            logger.info("Calling answer_question tool with response...")
            
            # Add timeout to prevent hanging
            await asyncio.wait_for(
                agent_tools["answer-question"].ainvoke({
                    "response": response
                }),
                timeout=30.0  # 30 second timeout
            )
            logger.info("Successfully sent response via runtime tool")
        except asyncio.TimeoutError:
            logger.error("Timeout sending response - UI connection may be broken")
        except Exception as e:
            logger.error(f"Failed to invoke answer_question tool: {str(e)}")
            raise
    else:
        logger.info("Interactive mode - response logged only (no runtime tool)")
        logger.info(f"Response: {response}")
    
    logger.info("Response sending completed")

async def main():
    """Main function to run the agent in a continuous loop with chat history."""
    try:
        config = load_config()
        
        # Build connection URL following reference pattern
        runtime = config["runtime"]
        base_url = config["coral_sse_url"] or os.getenv("CORAL_SSE_URL")
        agent_id = config["agent_id"] or os.getenv("CORAL_AGENT_ID")
        
        coral_params = {
            "agentId": agent_id,
            "agentDescription": "GeoSpot orchestrator that coordinates business location analysis between data analyzer and content generator agents"
        }
        
        query_string = urllib.parse.urlencode(coral_params)
        coral_server_url = f"{base_url}?{query_string}"
        
        logger.info(f"Connecting to Coral Server: {coral_server_url}")

        timeout = float(os.getenv("TIMEOUT_MS", "30000"))
        client = MultiServerMCPClient(
            connections={
                "coral": {
                    "transport": "sse",
                    "url": coral_server_url,
                    "timeout": timeout,
                    "sse_read_timeout": timeout,
                }
            }
        )
        logger.info("Coral Server connection established")

        coral_tools = await client.get_tools(server_name="coral")
        logger.info(f"Retrieved {len(coral_tools)} coral tools")

        # Check for runtime tools if needed
        if runtime is not None:
            required_tools = ["request-question", "answer-question"]
            available_tools = [tool.name for tool in coral_tools]
            
            for tool_name in required_tools:
                if tool_name not in available_tools:
                    error_message = f"Required tool '{tool_name}' not found in coral_tools"
                    logger.error(error_message)
                    raise ValueError(error_message)
        
        agent_tools = {tool.name: tool for tool in coral_tools}
        agent_executor = await create_agent(coral_tools)
        logger.info("Agent executor created")

        chat_history: List[Dict[str, str]] = []

        while True:
            try:
                user_input = await get_user_input(runtime, agent_tools)
                formatted_history = format_chat_history(chat_history)
                
                result = await agent_executor.ainvoke({
                    "user_input": user_input,
                    "chat_history": formatted_history
                })
                
                response = result.get('output', 'No output returned')
                await send_response(runtime, agent_tools, response)

                chat_history.append({"user_input": user_input, "response": response})
                if len(chat_history) > MAX_CHAT_HISTORY:
                    chat_history.pop(0)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")
                await asyncio.sleep(5)
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())