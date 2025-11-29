import os
import math
import time
import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CENSUS_ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
FCC_CENSUS_BLOCK_URL = "https://geo.fcc.gov/api/census/block/find"
AIML_URL = "https://api.aimlapi.com/v1/chat/completions"  # Switched from Mistral
ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

CENSUS_VARS = {
    "population_total": "B01001_001E",
    "median_income": "B19013_001E",
}

def geocode_bbox(place: str) -> Optional[Tuple[float, float, float, float]]:
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
    s, w, n, e = bbox
    return [(random.uniform(s, n), random.uniform(w, e)) for _ in range(k)]

def fcc_block_for_point(lat, lon):
    try:
        params = {"latitude": lat, "longitude": lon, "format": "json", "showall": "true"}
        r = requests.get(FCC_CENSUS_BLOCK_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning(f"FCC block lookup failed for ({lat}, {lon}): {e}")
    return None

def acs_demo_for_tract(state_fips, county_fips, tract_code, api_key):
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
        "population_total_median_tract": median(pop) if pop else float("nan"),
        "median_income_median_tract": median(inc) if inc else float("nan"),
    }

def generate_narrative_aiml(aiml_api_key, business, city, results):
    if not aiml_api_key:
        return "LLM summary skipped (no AIML API key)."
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
    headers = {"Authorization": f"Bearer {aiml_api_key}", "Content-Type": "application/json"}
    body = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",  # Use Mistral via AIML
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)}
        ],
        "temperature": 0.4,
        "max_tokens": 800
    }
    try:
        r = requests.post(AIML_URL, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        logger.info(f"AIML tokens used: {usage.get('total_tokens', 0)}")
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"AIML call failed: {e}")
        return "LLM summary unavailable."

def tts_elevenlabs(eleven_api_key, text, voice_id="Rachel"):
    if not eleven_api_key:
        return None
    try:
        headers = {
            "xi-api-key": eleven_api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {"text": text[:4000], "voice_settings": {"stability": 0.4, "similarity_boost": 0.9}}
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        out = "geospot_briefing.mp3"
        with open(out, "wb") as f:
            f.write(r.content)
        return out
    except requests.RequestException as e:
        logger.warning(f"ElevenLabs TTS failed: {e}")
    return None