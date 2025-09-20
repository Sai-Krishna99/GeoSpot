import math
from typing import Dict, Tuple

from .tools import overpass_query_pois, aggregate_demographics

def bbox_to_area(bbox: Tuple[float, float, float, float]) -> float:
    s, w, n, e = bbox
    R = 6371.0
    lat_km = math.pi * R / 180.0
    lon_km = lat_km * math.cos(math.radians((n + s) / 2.0))
    return max((n - s) * lat_km * (e - w) * lon_km, 1e-6)

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