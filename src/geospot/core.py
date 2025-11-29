
import json
import logging
from typing import List

from .models import BusinessProfile, NeighborhoodResult, GeoSpotRunConfig
from .tools import geocode_bbox, generate_narrative_aiml, tts_elevenlabs
from .scoring import compute_features, score_neighborhood

logger = logging.getLogger(__name__)

class CoralEngine:
    def __init__(self, cfg: GeoSpotRunConfig):
        self.cfg = cfg

    def run(self) -> List[NeighborhoodResult]:
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
        narrative = generate_narrative_aiml(
            self.cfg.aiml_api_key, self.cfg.business, self.cfg.city, top3
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
