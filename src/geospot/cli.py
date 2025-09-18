
import os
import argparse
from .core import (
    BusinessProfile, GeoSpotRunConfig, CoralEngine
)
from dotenv import load_dotenv

def app():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="geospot", description="GeoSpot — AI Location Scout")
    sub = parser.add_subparsers(dest="cmd")

    rank = sub.add_parser("rank", help="Rank neighborhoods")
    rank.add_argument("--city", type=str, default=os.getenv("GEOSPOT_CITY", "Austin, Texas"))
    rank.add_argument("--neighborhoods", type=str, default=os.getenv("GEOSPOT_NEIGHBORHOODS", "Downtown; East Austin; South Congress; Hyde Park"))
    rank.add_argument("--business-type", type=str, default="coffee_shop")
    rank.add_argument("--ideal-density", type=str, default="4000,15000", help="min,max ppl/km^2")
    rank.add_argument("--price", type=str, default="mid", choices=["value","mid","premium"])
    rank.add_argument("--foot-traffic", type=str, default="high", choices=["low","medium","high"])

    args = parser.parse_args()
    if args.cmd != "rank":
        parser.print_help()
        return

    ideal_min, ideal_max = [float(x.strip()) for x in args.ideal_density.split(",")]
    business = BusinessProfile(
        type=args.business_type,
        ideal_pop_density=(ideal_min, ideal_max),
        price_position=args.price,
        foot_traffic_importance=args.foot_traffic,
    )

    cfg = GeoSpotRunConfig(
        city=args.city,
        neighborhoods=[s.strip() for s in args.neighborhoods.split(";") if s.strip()],
        business=business,
        census_api_key=os.getenv("CENSUS_API_KEY"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        eleven_api_key=os.getenv("ELEVEN_API_KEY"),
        eleven_voice_id=os.getenv("ELEVEN_VOICE_ID", "Rachel"),
    )

    engine = CoralEngine(cfg)
    top3 = engine.run()

    print("\n=== GeoSpot — Top 3 Neighborhoods ===")
    for i, r in enumerate(top3, 1):
        print(f"{i}. {r.name}  |  Score: {r.score:.3f}")
        print(f"   Competition: {r.features['_counts_competition']}  |  Transit: {r.features['_counts_transit']}  |  Vibe: {r.features['_counts_vibe']}")
        print(f"   Median income (tract-median): ${int(r.features.get('median_income',0)):,}")
        if r.audio_path:
            print(f"   Audio briefing: {r.audio_path}")
        print()
    print("Full JSON report: geospot_report.json")
