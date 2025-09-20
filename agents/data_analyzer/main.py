import urllib.parse
from dotenv import load_dotenv
import os
import json
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
import requests
import math
import time
import random

from langchain.tools import Tool
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API URLs and constants
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CENSUS_ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
FCC_CENSUS_BLOCK_URL = "https://geo.fcc.gov/api/census/block/find"

CENSUS_VARS = {
    "population_total": "B01001_001E",
    "median_income": "B19013_001E",
}

def geocode_bbox(place: str) -> Optional[Tuple[float, float, float, float]]:
    """Geocode a place name to bounding box coordinates"""
    try:
        params = {"q": place, "format": "json", "limit": 1}
        r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": "GeoSpot/1.0"}, timeout=10)
        r.raise_for_status()
        if r.json():
            j = r.json()[0]
            return (float(j["boundingbox"][0]), float(j["boundingbox"][2]),
                    float(j["boundingbox"][1]), float(j["boundingbox"][3]))
    except requests.RequestException as e:
        logger.warning(f"Geocoding failed for {place}: {e}")
    return None

def overpass_query_pois(bbox, keys):
    """Query OpenStreetMap for points of interest in a bounding box"""
    if not keys:
        return []
    try:
        s, w, n, e = bbox
        filters = []
        for k in keys:
            filters.extend([
                f'node["amenity"="{k}"]({s},{w},{n},{e});',
                f'node["shop"="{k}"]({s},{w},{n},{e});',
                f'node["leisure"="{k}"]({s},{w},{n},{e});',
                f'node["tourism"="{k}"]({s},{w},{n},{e});'
            ])
        q = f"""
        [out:json][timeout:25];
        (
          {"".join(filters)}
        );
        out center;
        """
        r = requests.post(OVERPASS_URL, data=q.encode("utf-8"), timeout=30)
        r.raise_for_status()
        return r.json().get("elements", [])
    except requests.RequestException as e:
        logger.warning(f"Overpass query failed: {e}")
        return []

def sample_points_in_bbox(bbox, k=12):
    """Generate random sample points within a bounding box"""
    s, w, n, e = bbox
    return [(random.uniform(s, n), random.uniform(w, e)) for _ in range(k)]

def fcc_block_for_point(lat, lon):
    """Get census block information for a geographic point"""
    try:
        params = {"latitude": lat, "longitude": lon, "format": "json", "showall": "true"}
        r = requests.get(FCC_CENSUS_BLOCK_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning(f"FCC block lookup failed for ({lat}, {lon}): {e}")
    return None

def acs_demo_for_tract(state_fips, county_fips, tract_code, api_key):
    """Get demographic data for a census tract"""
    try:
        vars_str = ",".join(["NAME"] + list(CENSUS_VARS.values()))
        params = {
            "get": vars_str,
            "for": f"tract:{tract_code}",
            "in": f"state:{state_fips} county:{county_fips}"
        }
        if api_key:
            params["key"] = api_key
        r = requests.get(CENSUS_ACS_URL, params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if len(rows) >= 2:
            headers = rows[0]
            values = rows[1]
            return dict(zip(headers, values))
    except requests.RequestException as e:
        logger.warning(f"Census ACS query failed: {e}")
    return None

def aggregate_demographics(bbox, api_key):
    """Aggregate demographic data for a bounding box"""
    pts = sample_points_in_bbox(bbox, k=12)
    pop, inc = [], []
    for lat, lon in pts:
        blk = fcc_block_for_point(lat, lon)
        if not blk or "County" not in blk or "State" not in blk or "Block" not in blk:
            continue
        state_fips = blk["State"]["FIPS"]
        county_fips = blk["County"]["FIPS"]
        full = blk["Block"]["FIPS"]
        tract_code = full[5:11]
        demo = acs_demo_for_tract(state_fips, county_fips, tract_code, api_key)
        if not demo:
            continue
        try:
            p = float(demo.get(CENSUS_VARS["population_total"], "nan"))
            mhi = float(demo.get(CENSUS_VARS["median_income"], "nan"))
            if math.isfinite(p):
                pop.append(p)
            if math.isfinite(mhi) and mhi > 0:
                inc.append(mhi)
        except Exception as e:
            logger.warning(f"Error parsing demographics: {e}")
            continue
        time.sleep(0.15)
    
    def median(xs):
        if not xs: return float("nan")
        x = sorted(xs)
        m = len(x)//2
        return x[m] if len(x)%2 else 0.5*(x[m-1]+x[m])
    
    return {
        "population_total": median(pop) if pop else float("nan"),
        "median_income": median(inc) if inc else float("nan"),
    }

def compute_features(bbox, business_type, census_api_key):
    """Compute all features for a neighborhood"""
    features = {}
    
    # Define POI types based on business type
    if business_type.lower() in ['coffee', 'cafe', 'coffeeshop']:
        competition_keys = ['cafe', 'restaurant']
        vibe_keys = ['library', 'bookshop', 'art', 'museum']
        transit_keys = ['bus_station', 'subway_entrance', 'tram_stop']
    elif business_type.lower() in ['bookstore', 'book']:
        competition_keys = ['bookshop', 'library']
        vibe_keys = ['cafe', 'university', 'school', 'museum']
        transit_keys = ['bus_station', 'subway_entrance', 'tram_stop']
    else:
        # Default for other business types
        competition_keys = ['shop']
        vibe_keys = ['cafe', 'restaurant', 'bar']
        transit_keys = ['bus_station', 'subway_entrance', 'tram_stop']
    
    # Get POI counts
    competition_pois = overpass_query_pois(bbox, competition_keys)
    vibe_pois = overpass_query_pois(bbox, vibe_keys)
    transit_pois = overpass_query_pois(bbox, transit_keys)
    
    features['_counts_competition'] = len(competition_pois)
    features['_counts_vibe'] = len(vibe_pois)
    features['_counts_transit'] = len(transit_pois)
    
    # Get demographics
    demographics = aggregate_demographics(bbox, census_api_key)
    features.update(demographics)
    
    return features

def score_neighborhood(features, business_type):
    """Score a neighborhood based on features"""
    score = 0.0
    
    # Competition scoring (moderate competition is good)
    competition = features.get('_counts_competition', 0)
    if competition == 0:
        score += 0.2  # No competition might mean no market
    elif 1 <= competition <= 5:
        score += 0.8  # Good balance
    elif 6 <= competition <= 10:
        score += 0.6  # Moderate competition
    else:
        score += 0.3  # Too much competition
    
    # Vibe scoring (more is better)
    vibe = features.get('_counts_vibe', 0)
    score += min(vibe * 0.1, 0.8)  # Cap at 0.8
    
    # Transit scoring (more is better)
    transit = features.get('_counts_transit', 0)
    score += min(transit * 0.15, 0.7)  # Cap at 0.7
    
    # Income scoring
    income = features.get('median_income', 0)
    if math.isfinite(income) and income > 0:
        # Normalize income (50k = 0.5, 100k = 1.0)
        income_score = min(income / 100000, 1.0)
        score += income_score * 0.5
    
    # Population density
    population = features.get('population_total', 0)
    if math.isfinite(population) and population > 0:
        # Normalize population (higher is generally better)
        pop_score = min(population / 5000, 1.0)
        score += pop_score * 0.3
    
    return min(score, 5.0)  # Cap at 5.0

# ==== EXTERNAL API TOOL FUNCTIONS ====
# These functions are exposed as callable tools for the AIML API

def analyze_location_tool(city: str, business_type: str) -> str:
    """
    Analyze neighborhoods in a city for business suitability.
    
    Args:
        city: The city to analyze (e.g., "Austin, Texas")
        business_type: Type of business (e.g., "coffee shop", "bookstore")
    
    Returns:
        JSON string with neighborhood analysis and scores
    """
    try:
        logger.info(f"Starting location analysis for {business_type} in {city}")
        
        # Geocode the city to get bounding box
        bbox = geocode_bbox(city)
        if not bbox:
            return json.dumps({
                "error": f"Could not geocode city: {city}",
                "recommendations": []
            })
        
        logger.info(f"Geocoded {city} to bbox: {bbox}")
        
        # Get census API key from config
        census_key = os.getenv("CENSUS_API_KEY", "")
        
        # Compute features for the area
        features = compute_features(bbox, business_type, census_key)
        score = score_neighborhood(features, business_type)
        
        # Prepare results
        analysis_result = {
            "city": city,
            "business_type": business_type,
            "bbox": bbox,
            "overall_score": round(score, 2),
            "features": {
                "competition_count": features.get('_counts_competition', 0),
                "vibe_locations": features.get('_counts_vibe', 0),
                "transit_access": features.get('_counts_transit', 0),
                "median_income": features.get('median_income', 0),
                "population": features.get('population_total', 0)
            },
            "recommendations": []
        }
        
        # Generate business-specific recommendations
        if business_type.lower() in ['coffee', 'cafe', 'coffeeshop']:
            if score >= 4.0:
                analysis_result["recommendations"] = [
                    f"Excellent location for a coffee shop! Score: {score:.1f}/5.0",
                    f"Found {features.get('_counts_competition', 0)} competing coffee establishments - good market validation",
                    f"Strong foot traffic indicators with {features.get('_counts_vibe', 0)} complementary businesses nearby",
                    f"Good transit access with {features.get('_counts_transit', 0)} transportation options"
                ]
            elif score >= 3.0:
                analysis_result["recommendations"] = [
                    f"Good potential for coffee shop. Score: {score:.1f}/5.0",
                    f"Moderate competition with {features.get('_counts_competition', 0)} existing establishments",
                    f"Consider location near high foot traffic areas"
                ]
            else:
                analysis_result["recommendations"] = [
                    f"Challenging location for coffee shop. Score: {score:.1f}/5.0",
                    "Consider exploring other neighborhoods with better foot traffic",
                    "Market research recommended before proceeding"
                ]
        
        logger.info(f"Analysis completed. Overall score: {score:.2f}")
        return json.dumps(analysis_result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in location analysis: {str(e)}")
        return json.dumps({
            "error": f"Analysis failed: {str(e)}",
            "recommendations": ["Please try again or contact support"]
        })

def get_neighborhood_details_tool(city: str, neighborhood: str) -> str:
    """
    Get detailed analysis for a specific neighborhood.
    
    Args:
        city: The city name
        neighborhood: Specific neighborhood to analyze
    
    Returns:
        JSON string with detailed neighborhood metrics
    """
    try:
        location_query = f"{neighborhood}, {city}"
        logger.info(f"Getting details for {location_query}")
        
        bbox = geocode_bbox(location_query)
        if not bbox:
            return json.dumps({
                "error": f"Could not find neighborhood: {location_query}"
            })
        
        # Get POI data for different categories
        competition_pois = overpass_query_pois(bbox, ['cafe', 'restaurant'])
        vibe_pois = overpass_query_pois(bbox, ['library', 'bookshop', 'art', 'museum'])
        transit_pois = overpass_query_pois(bbox, ['bus_station', 'subway_entrance'])
        
        census_key = os.getenv("CENSUS_API_KEY", "")
        demographics = aggregate_demographics(bbox, census_key)
        
        result = {
            "neighborhood": neighborhood,
            "city": city,
            "bbox": bbox,
            "poi_analysis": {
                "competition_locations": len(competition_pois),
                "cultural_venues": len(vibe_pois),
                "transit_options": len(transit_pois)
            },
            "demographics": {
                "median_income": demographics.get('median_income', 0),
                "population": demographics.get('population_total', 0)
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting neighborhood details: {str(e)}")
        return json.dumps({"error": str(e)})

def get_tools_description(tools):
    """Get description of available tools"""
    descriptions = []
    for tool in tools:
        if hasattr(tool, 'args'):
            # Coral tools have args attribute
            schema = json.dumps(tool.args).replace('{', '{{').replace('}', '}}')
            descriptions.append(f"Tool: {tool.name}, Schema: {schema}")
        else:
            # Analysis tools don't have args, use description
            descriptions.append(f"Tool: {tool.name}, Description: {tool.description}")
    return "\n".join(descriptions)

def load_config():
    """Load configuration from environment"""
    logger.info("Starting configuration loading...")
    
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME")
    logger.info(f"CORAL_ORCHESTRATION_RUNTIME: {runtime}")
    
    if runtime is None:
        logger.info("Runtime not found, loading .env file...")
        load_dotenv()
        logger.info(".env file loaded successfully")
    
    logger.info("Building configuration dictionary...")
    config = {
        "runtime": runtime,
        "agent_id": os.getenv("CORAL_AGENT_ID", "geospot-data-analyzer"),
        "model_name": "openai/gpt-4.1-2025-04-14",
        "model_provider": "aiml", 
        "api_key": os.getenv("MODEL_API_KEY", ""),
        "census_api_key": os.getenv("CENSUS_API_KEY", "")
    }
    
    logger.info("Configuration loaded:")
    logger.info(f"  - runtime: {config['runtime']}")
    logger.info(f"  - agent_id: {config['agent_id']}")
    logger.info(f"  - model_name: {config['model_name']}")
    logger.info(f"  - model_provider: {config['model_provider']}")
    logger.info(f"  - api_key: {'***' if config['api_key'] else 'NOT SET'}")
    
    logger.info("Validating required fields...")
    required_fields = ["agent_id", "model_name", "api_key"]
    for field in required_fields:
        if not config.get(field):
            raise ValueError(f"Required configuration field '{field}' is missing")
    
    logger.info("All required fields present")
    return config

def create_analysis_tools():
    """Create analysis tools for external API integration using proper LangChain tool format"""
    tools = []
    
    # Location analysis tool
    analyze_tool = Tool(
        name="analyze_location",
        description="Analyze neighborhoods in a city for business suitability using external APIs (OpenStreetMap, Census data). Input should be: city and business_type separated by comma.",
        func=lambda query: analyze_location_wrapper(query)
    )
    tools.append(analyze_tool)
    
    # Neighborhood details tool  
    details_tool = Tool(
        name="get_neighborhood_details",
        description="Get detailed analysis for a specific neighborhood including POI data and demographics. Input should be: city and neighborhood separated by comma.",
        func=lambda query: neighborhood_details_wrapper(query)
    )
    tools.append(details_tool)
    
    return tools

def analyze_location_wrapper(query: str) -> str:
    """Wrapper to parse query string for analyze_location_tool"""
    try:
        parts = query.split(',')
        if len(parts) >= 2:
            city = parts[0].strip()
            business_type = parts[1].strip()
            return analyze_location_tool(city, business_type)
        else:
            return "Please provide both city and business_type separated by comma"
    except Exception as e:
        logger.error(f"Error in analyze_location_wrapper: {str(e)}")
        return f"Analysis error: {str(e)}"

def neighborhood_details_wrapper(query: str) -> str:
    """Wrapper to parse query string for get_neighborhood_details_tool"""
    try:
        parts = query.split(',')
        if len(parts) >= 2:
            city = parts[0].strip()
            neighborhood = parts[1].strip()
            return get_neighborhood_details_tool(city, neighborhood)
        else:
            return "Please provide both city and neighborhood separated by comma"
    except Exception as e:
        logger.error(f"Error in neighborhood_details_wrapper: {str(e)}")
        return f"Details error: {str(e)}"

async def create_agent(coral_tools, analysis_tools):
    coral_tools_description = get_tools_description(coral_tools)
    analysis_tools_description = get_tools_description(analysis_tools)
    combined_tools = coral_tools + analysis_tools
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are the GeoSpot Data Analyzer Agent interacting with Coral Server tools and having your own location analysis tools. Your task is to perform location analysis instructions from other agents.

Follow these steps in order:
1. Call wait_for_mentions from coral tools (timeoutMs: 30000) to receive mentions from other agents.
2. When you receive a mention, keep the thread ID and the sender ID.
3. Take 2 seconds to think about the content (instruction) of the message and extract:
   - Business type (e.g., "coffee shop", "bookstore")
   - City/location (e.g., "Plano, TX", "Austin, Texas")
   - Any specific neighborhoods mentioned
4. Check the tool schema and make a plan for location analysis:
   - Use analyze_location tool with city and business_type
   - If specific neighborhoods mentioned, use get_neighborhood_details
   - Generate comprehensive analysis with scores and recommendations
5. Execute your analysis tools to gather location data, demographics, and business suitability scores.
6. Take 3 seconds and think about the analysis results. Create a comprehensive response with:
   - Overall neighborhood scores and explanations
   - Specific metrics (competition, demographics, transit access)
   - Business-specific recommendations
   - Data-driven insights for decision making
7. Use send_message from coral tools to send the analysis results to the sender ID in the same thread ID.
8. If any error occurs, use send_message to send error details to the sender.
9. Always respond back to the sender agent even if analysis fails.
10. Wait for 2 seconds and repeat from step 1.

These are the coral tools: {coral_tools_description}
These are your analysis tools: {analysis_tools_description}"""
        ),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model="openai/gpt-4.1-2025-04-14",
        model_provider="openai",
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=0.1,
        max_tokens=2000,
        base_url="https://api.aimlapi.com/v1"
    )
    
    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "GeoSpot data analyzer that fetches location data, demographics, and scores neighborhoods for business suitability"
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
    analysis_tools = create_analysis_tools()
    print(f"Coral tools count: {len(coral_tools)} and analysis tools count: {len(analysis_tools)}")

    agent_executor = await create_agent(coral_tools, analysis_tools)

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