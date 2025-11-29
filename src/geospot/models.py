from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_POI = {
    "competition": ["cafe", "coffee_shop", "restaurant"],
    "transit": ["bus_station", "tram_stop", "train_station", "subway_entrance"],
    "vibe": ["bar", "pub", "nightclub", "art_gallery", "theatre", "cinema", "music_venue", "park", "bookshop", "library"]
}

DEFAULT_WEIGHTS = {
    "competition_density": -0.30,
    "population_density":  0.25,
    "median_income":       0.20,
    "education_proxy":     0.10,
    "transit_access":      0.10,
    "vibe_mix":            0.05
}

@dataclass
class BusinessProfile:
    type: str
    ideal_pop_density: tuple
    price_position: str
    foot_traffic_importance: str

@dataclass
class NeighborhoodResult:
    name: str
    bbox: tuple
    features: Dict[str, float]
    score: float
    rationale: str = ""
    audio_path: Optional[str] = None

@dataclass
class GeoSpotRunConfig:
    city: str
    neighborhoods: List[str]
    business: BusinessProfile
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS)
    poi_spec: Dict[str, List[str]] = field(default_factory=lambda: DEFAULT_POI)
    census_api_key: Optional[str] = None
    aiml_api_key: Optional[str] = None  # Switched from mistral
    eleven_api_key: Optional[str] = None
    eleven_voice_id: str = "Rachel"