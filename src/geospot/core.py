
import os
import math
import time
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CENSUS_ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
FCC_CENSUS_BLOCK_URL = "https://geo.fcc.gov/api/census/block/find"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

CENSUS_VARS = {
    "population_total": "B01001_001E",
    "median_income": "B19013_001E",
}

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
    mistral_api_key: Optional[str] = None
    eleven_api_key: Optional[str] = None
    eleven_voice_id: str = "Rachel"

def geocode_bbox(place: str) -> Optional[Tuple[float, float, float, float]]:
    params = {"q": place, "format": "json", "limit": 1}
    r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": "GeoSpot/1.0"})
    if r.ok and r.json():
        j = r.json()[0]
        return (float(j["boundingbox"][0]), float(j["boundingbox"][2]),
                float(j["boundingbox"][1]), float(j["boundingbox"][3]))
    return None

def bbox_to_area(bbox: Tuple[float, float, float, float]) -> float:
    s, w, n, e = bbox
    R = 6371.0
    lat_km = math.pi * R / 180.0
    lon_km = lat_km * math.cos(math.radians((n + s) / 2.0))
    return max((n - s) * lat_km * (e - w) * lon_km, 1e-6)

def overpass_query_pois(bbox, keys):
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
    r = requests.post(OVERPASS_URL, data=q.encode("utf-8"))
    if not r.ok:
        return []
    return r.json().get("elements", [])

def sample_points_in_bbox(bbox, k=12):
    s, w, n, e = bbox
    return [(random.uniform(s, n), random.uniform(w, e)) for _ in range(k)]

def fcc_block_for_point(lat, lon):
    params = {"latitude": lat, "longitude": lon, "format": "json", "showall": "true"}
    r = requests.get(FCC_CENSUS_BLOCK_URL, params=params)
    if r.ok:
        return r.json()
    return None

def acs_demo_for_tract(state_fips, county_fips, tract_code, api_key):
    vars_str = ",".join(["NAME"] + list(CENSUS_VARS.values()))
    params = {
        "get": vars_str,
        "for": f"tract:{tract_code}",
        "in": f"state:{state_fips} county:{county_fips}"
    }
    if api_key:
        params["key"] = api_key
    r = requests.get(CENSUS_ACS_URL, params=params)
    if not r.ok:
        return None
    rows = r.json()
    if len(rows) >= 2:
        headers = rows[0]
        values = rows[1]
        return dict(zip(headers, values))
    return None

def aggregate_demographics(bbox, api_key):
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
        except Exception:
            continue
        time.sleep(0.15)
    def median(xs):
        if not xs: return float("nan")
        x = sorted(xs)
        m = len(x)//2
        return x[m] if len(x)%2 else 0.5*(x[m-1]+x[m])
    return {
        "population_total_median_tract": median(pop) if pop else float("nan"),
        "median_income_median_tract": median(inc) if inc else float("nan"),
    }

def normalize(value, ref_low, ref_high, invert=False):
    if any(map(lambda v: not math.isfinite(v), [value, ref_low, ref_high])) or ref_high <= ref_low:
        return 0.5
    x = (value - ref_low) / (ref_high - ref_low)
    x = max(0.0, min(1.0, x))
    return 1.0 - x if invert else x

def shannon_entropy(buckets):
    tot = sum(buckets.values()) or 1
    ent = 0.0
    for c in buckets.values():
        p = c / tot
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    k = len(buckets) or 1
    return ent / math.log(k + 1e-12)

def compute_features(bbox, poi_spec, census_api_key, business):
    area_km2 = bbox_to_area(bbox)
    competition_nodes = overpass_query_pois(bbox, poi_spec["competition"])
    comp_density = len(competition_nodes) / area_km2
    transit_nodes = overpass_query_pois(bbox, poi_spec["transit"])
    transit_per_km2 = len(transit_nodes) / area_km2
    vibe_nodes = overpass_query_pois(bbox, poi_spec["vibe"])
    vibe_counts = {}
    for n in vibe_nodes:
        tags = n.get("tags", {})
        for f in ("amenity","shop","leisure","tourism"):
            if f in tags:
                vibe_counts[tags[f]] = vibe_counts.get(tags[f], 0) + 1
                break
    vibe_mix = shannon_entropy(vibe_counts)
    demo = aggregate_demographics(bbox, census_api_key)
    pop_median_tract = demo.get("population_total_median_tract", float("nan"))
    mhi_median_tract = demo.get("median_income_median_tract", float("nan"))
    pop_density = pop_median_tract / max(area_km2, 1e-6) if math.isfinite(pop_median_tract) else float("nan")
    f_comp = normalize(comp_density, 0, 80, invert=True)
    f_transit = normalize(transit_per_km2, 0, 30)
    ideal_min, ideal_max = business.ideal_pop_density
    if math.isfinite(pop_density):
        if pop_density < ideal_min:
            f_pop = normalize(pop_density, 0, ideal_min) * 0.7
        elif pop_density > ideal_max:
            f_pop = normalize(pop_density, ideal_max, ideal_max*2.0, invert=True) * 0.7
        else:
            f_pop = 0.8 + 0.2 * normalize(pop_density, ideal_min, ideal_max)
    else:
        f_pop = 0.5
    f_mhi = normalize(mhi_median_tract if math.isfinite(mhi_median_tract) else 60000, 30000, 110000)
    f_vibe = normalize(vibe_mix, 0.0, 1.0)
    foot_mult = {"low": 0.8, "medium": 1.0, "high": 1.2}.get(business.foot_traffic_importance, 1.0)
    features = {
        "competition_density": comp_density,
        "population_density": pop_density if math.isfinite(pop_density) else 0.0,
        "median_income": mhi_median_tract if math.isfinite(mhi_median_tract) else 0.0,
        "education_proxy": 0.0,
        "transit_access": transit_per_km2,
        "vibe_mix": vibe_mix,
        "_norm_competition": f_comp,
        "_norm_population": f_pop,
        "_norm_income": f_mhi,
        "_norm_education": 0.5,
        "_norm_transit": min(1.0, f_transit * foot_mult),
        "_norm_vibe": f_vibe,
        "_area_km2": area_km2,
        "_counts_competition": len(competition_nodes),
        "_counts_transit": len(transit_nodes),
        "_counts_vibe": len(vibe_nodes),
    }
    return features

def score_neighborhood(features, weights):
    return (
        weights.get("competition_density", -0.3) * features["_norm_competition"] +
        weights.get("population_density", 0.25)   * features["_norm_population"] +
        weights.get("median_income", 0.2)         * features["_norm_income"] +
        weights.get("education_proxy", 0.1)       * features["_norm_education"] +
        weights.get("transit_access", 0.1)        * features["_norm_transit"] +
        weights.get("vibe_mix", 0.05)             * features["_norm_vibe"]
    )

def generate_narrative_mistral(mistral_api_key, business, city, results):
    if not mistral_api_key:
        return "LLM summary skipped (no Mistral API key)."
    sys_prompt = (
        "You are a world-class retail site selection strategist. "
        "Given scored features for neighborhoods, write a crisp, persuasive briefing: "
        "1) Rank-ordered picks with 2-3 bullet pros/cons each, "
        "2) A one-sentence recommendation, "
        "3) Assumptions/limits in one short line."
    )
    user_payload = {
        "business": business.__dict__,
        "city": city,
        "results": [
            {"name": r.name, "score": round(r.score, 3), "features": r.features}
            for r in results
        ]
    }
    headers = {"Authorization": f"Bearer {mistral_api_key}"}
    body = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)}
        ],
        "temperature": 0.4,
        "max_tokens": 800
    }
    r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=body, timeout=60)
    if not r.ok:
        return f"Mistral call failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return "LLM summary unavailable."

def tts_elevenlabs(eleven_api_key, text, voice_id="Rachel"):
    if not eleven_api_key:
        return None
    headers = {
        "xi-api-key": eleven_api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {"text": text[:4000], "voice_settings": {"stability": 0.4, "similarity_boost": 0.9}}
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    r = requests.post(url, headers=headers, json=payload)
    if not r.ok:
        return None
    out = "geospot_briefing.mp3"
    with open(out, "wb") as f:
        f.write(r.content)
    return out

class CoralEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        city_bbox = geocode_bbox(self.cfg.city)
        if not city_bbox:
            raise RuntimeError(f"Could not geocode city: {self.cfg.city}")
        results = []
        for nb in self.cfg.neighborhoods:
            bbox = geocode_bbox(f"{nb}, {self.cfg.city}") or city_bbox
            feats = compute_features(bbox, self.cfg.poi_spec, self.cfg.census_api_key, self.cfg.business)
            score = score_neighborhood(feats, self.cfg.weights)
            results.append(NeighborhoodResult(name=nb, bbox=bbox, features=feats, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        top3 = results[:3]
        narrative = generate_narrative_mistral(
            self.cfg.mistral_api_key, self.cfg.business, self.cfg.city, top3
        )
        for r in top3:
            r.rationale = f"{r.name}: score {r.score:.3f}. Key signals → comp:{r.features['_counts_competition']} / transit:{r.features['_counts_transit']} / vibe:{r.features['_counts_vibe']} / median_income:${int(r.features.get('median_income',0))}"
        audio_text = (f"GeoSpot briefing for {self.cfg.city}. Top pick: {top3[0].name}. {narrative[:800]}")
        audio_path = tts_elevenlabs(self.cfg.eleven_api_key, audio_text, self.cfg.eleven_voice_id)
        if audio_path:
            top3[0].audio_path = audio_path
        report = {
            "city": self.cfg.city,
            "business": self.cfg.business.__dict__,
            "ranked": [
                {
                    "name": r.name,
                    "score": r.score,
                    "bbox": r.bbox,
                    "features": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.features.items()},
                    "rationale": r.rationale,
                    "audio_path": r.audio_path,
                }
                for r in top3
            ],
            "narrative": narrative
        }
        with open("geospot_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return top3
