
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import os

from ..core import BusinessProfile, GeoSpotRunConfig, CoralEngine

app = FastAPI(title="GeoSpot API", description="AI Location Scout as a service")

class RankRequest(BaseModel):
    city: str
    neighborhoods: List[str]
    business_type: str = "coffee_shop"
    ideal_density: str = "4000,15000"
    price: str = "mid"
    foot_traffic: str = "high"

@app.post("/rank")
def rank(req: RankRequest):
    ideal_min, ideal_max = [float(x.strip()) for x in req.ideal_density.split(",")]
    business = BusinessProfile(
        type=req.business_type,
        ideal_pop_density=(ideal_min, ideal_max),
        price_position=req.price,
        foot_traffic_importance=req.foot_traffic,
    )

    cfg = GeoSpotRunConfig(
        city=req.city,
        neighborhoods=req.neighborhoods,
        business=business,
        census_api_key=os.getenv("CENSUS_API_KEY"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        eleven_api_key=os.getenv("ELEVEN_API_KEY"),
        eleven_voice_id=os.getenv("ELEVEN_VOICE_ID", "Rachel"),
    )

    engine = CoralEngine(cfg)
    top3 = engine.run()
    return {
        "city": cfg.city,
        "results": [
            {
                "name": r.name,
                "score": r.score,
                "features": r.features,
                "rationale": r.rationale,
                "audio_path": r.audio_path,
            }
            for r in top3
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
