
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI
from pydantic import BaseModel, Field, conlist
import os

from .core import (
    BusinessProfile, GeoSpotRunConfig, CoralEngine
)

app = FastAPI(title="GeoSpot API", version="0.1")

class BusinessModel(BaseModel):
    type: str = Field("coffee_shop", description="Business type, e.g., coffee_shop, bookstore")
    ideal_pop_density: conlist(float, min_items=2, max_items=2) = Field([4000, 15000], description="min,max ppl per km^2")
    price_position: str = Field("mid", description="value | mid | premium")
    foot_traffic_importance: str = Field("high", description="low | medium | high")

class RankRequest(BaseModel):
    city: Optional[str] = Field(None, description="City (falls back to env GEOSPOT_CITY if omitted)")
    neighborhoods: Optional[List[str]] = Field(None, description="Neighborhoods (falls back to env GEOSPOT_NEIGHBORHOODS if omitted)")
    business: BusinessModel = BusinessModel()

class NeighborhoodOut(BaseModel):
    name: str
    score: float
    bbox: Tuple[float, float, float, float]
    features: Dict[str, Any]
    rationale: str
    audio_path: Optional[str] = None

class RankResponse(BaseModel):
    city: str
    ranked: List[NeighborhoodOut]
    narrative: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    city = req.city or os.getenv("GEOSPOT_CITY", "Austin, Texas")
    neighborhoods = req.neighborhoods
    if neighborhoods is None:
        neighborhoods_env = os.getenv("GEOSPOT_NEIGHBORHOODS", "Downtown; East Austin; South Congress; Hyde Park")
        neighborhoods = [s.strip() for s in neighborhoods_env.split(";") if s.strip()]

    business = BusinessProfile(
        type=req.business.type,
        ideal_pop_density=tuple(req.business.ideal_pop_density),
        price_position=req.business.price_position,
        foot_traffic_importance=req.business.foot_traffic_importance,
    )

    cfg = GeoSpotRunConfig(
        city=city,
        neighborhoods=neighborhoods,
        business=business,
        census_api_key=os.getenv("CENSUS_API_KEY"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        eleven_api_key=os.getenv("ELEVEN_API_KEY"),
        eleven_voice_id=os.getenv("ELEVEN_VOICE_ID", "Rachel"),
    )

    engine = CoralEngine(cfg)
    top3 = engine.run()

    ranked = [NeighborhoodOut(
        name=r.name,
        score=r.score,
        bbox=r.bbox,
        features=r.features,
        rationale=r.rationale,
        audio_path=r.audio_path
    ) for r in top3]

    # Narrative is stored alongside in report; recompute via the same json or leave implicit.
    # For simplicity, we'll read it back from geospot_report.json
    narrative = ""
    try:
        import json
        with open("geospot_report.json", "r", encoding="utf-8") as f:
            narrative = json.load(f).get("narrative", "")
    except Exception:
        narrative = ""

    return RankResponse(city=city, ranked=ranked, narrative=narrative)

def serve():
    import uvicorn
    uvicorn.run("geospot.server:app", host="0.0.0.0", port=8000, reload=False)
