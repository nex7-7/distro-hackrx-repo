"""Riddle Solver Module

Implements special branching logic:
1. Fetch favourite city from the HackRx endpoint.
2. Map the returned city to a landmark using the supplied (intentionally shuffled) tables.
3. Based on the landmark, call one of five flight number endpoints.
4. Return a human‑readable string containing the flight number.

Design notes:
- Duplicate city entries across the two tables are resolved by keeping the first
  encountered landmark for a city (order preserves given listing). This is
  sufficient because flight resolution depends only on four landmark names.
- All network operations use an injected httpx.AsyncClient to reuse connections
  managed by the caller (FastAPI endpoint).
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import httpx
import asyncio

FAVOURITE_CITY_ENDPOINT = "https://register.hackrx.in/submissions/myFavouriteCity"

# Flight number endpoints (selection depends on resolved landmark)
FLIGHT_ENDPOINTS = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber",
    "DEFAULT": "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber",
}

# Raw (Landmark, Current Location) pairs from the two provided tables.
_LANDMARK_CITY_PAIRS: List[Tuple[str, str]] = [
    # Table 1
    ("Gateway of India", "Delhi"),
    ("India Gate", "Mumbai"),
    ("Charminar", "Chennai"),
    ("Marina Beach", "Hyderabad"),
    ("Howrah Bridge", "Ahmedabad"),
    ("Golconda Fort", "Mysuru"),
    ("Qutub Minar", "Kochi"),
    ("Taj Mahal", "Hyderabad"),
    ("Meenakshi Temple", "Pune"),
    ("Lotus Temple", "Nagpur"),
    ("Mysore Palace", "Chandigarh"),
    ("Rock Garden", "Kerala"),
    ("Victoria Memorial", "Bhopal"),
    ("Vidhana Soudha", "Varanasi"),
    ("Sun Temple", "Jaisalmer"),
    ("Golden Temple", "Pune"),
    # Table 2
    ("Eiffel Tower", "New York"),
    ("Statue of Liberty", "London"),
    ("Big Ben", "Tokyo"),
    ("Colosseum", "Beijing"),
    ("Sydney Opera House", "London"),
    ("Christ the Redeemer", "Bangkok"),
    ("Burj Khalifa", "Toronto"),
    ("CN Tower", "Dubai"),
    ("Petronas Towers", "Amsterdam"),
    ("Leaning Tower of Pisa", "Cairo"),
    ("Mount Fuji", "San Francisco"),
    ("Niagara Falls", "Berlin"),
    ("Louvre Museum", "Barcelona"),
    ("Stonehenge", "Moscow"),
    ("Sagrada Familia", "Seoul"),
    ("Acropolis", "Cape Town"),
    ("Big Ben", "Istanbul"),
    ("Machu Picchu", "Riyadh"),
    ("Taj Mahal", "Paris"),
    ("Moai Statues", "Dubai Airport"),
    ("Christchurch Cathedral", "Singapore"),
    ("The Shard", "Jakarta"),
    ("Blue Mosque", "Vienna"),
    ("Neuschwanstein Castle", "Kathmandu"),
    ("Buckingham Palace", "Los Angeles"),
    ("Space Needle", "Mumbai"),
    ("Times Square", "Seoul"),
]

def _build_city_to_landmark() -> Dict[str, str]:
    """Create a mapping from City -> Landmark.

    If a city appears multiple times, the first landmark encountered is kept
    (stable mapping) to avoid arbitrary overwrites.
    """
    mapping: Dict[str, str] = {}
    for landmark, city in _LANDMARK_CITY_PAIRS:
        if city not in mapping:  # Preserve first occurrence
            mapping[city] = landmark
    return mapping

CITY_TO_LANDMARK = _build_city_to_landmark()

async def solve_riddle(http_client: httpx.AsyncClient) -> str:
    """Execute the riddle solving workflow and return the final response string.

    Steps:
      1. Fetch favourite city.
      2. Resolve city -> landmark.
      3. Choose flight endpoint based on landmark (four specific cases + default).
      4. Fetch flight number and format final answer string.
    """
    # 1. Get favourite city
    fav_resp = await http_client.get(FAVOURITE_CITY_ENDPOINT, timeout=15.0)
    fav_resp.raise_for_status()
    fav_json = fav_resp.json()
    city = fav_json.get("data", {}).get("city")
    if not city:
        raise ValueError("City not found in favourite city response")

    # 2. Map city -> landmark
    landmark = CITY_TO_LANDMARK.get(city)
    if not landmark:
        landmark = "UNKNOWN"  # Fallback

    # 3. Select flight endpoint
    if landmark in FLIGHT_ENDPOINTS:
        flight_url = FLIGHT_ENDPOINTS[landmark]
    else:
        flight_url = FLIGHT_ENDPOINTS["DEFAULT"]

    # 4. Fetch flight number
    flight_resp = await http_client.get(flight_url, timeout=15.0)
    flight_resp.raise_for_status()
    flight_json = flight_resp.json()
    flight_number = flight_json.get("data", {}).get("flightNumber")
    if not flight_number:
        raise ValueError("Flight number not present in flight endpoint response")
    # Wait for 4 seconds before sending the response (per user request)
    await asyncio.sleep(4)
    return f"The flight number is {flight_number}"

__all__ = ["solve_riddle", "CITY_TO_LANDMARK"]
