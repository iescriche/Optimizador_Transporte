import json
import os
import hashlib
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import streamlit as st

CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "matrices.json")
CACHE_EXPIRY_DAYS = 30

def get_cache_key(coords: List[Tuple[float, float]], mode: str) -> str:
    """Generate a hash key for the cache based on coordinates and mode."""
    # Round coordinates to 6 decimal places for consistency
    rounded_coords = [(round(lat, 6), round(lon, 6)) for lat, lon in coords]
    coords_str = json.dumps(sorted(rounded_coords), separators=(",", ":"))
    mode_str = mode
    return hashlib.sha256(f"{coords_str}{mode_str}".encode()).hexdigest()

def load_cached_matrices(coords: List[Tuple[float, float]], mode: str) -> Optional[Tuple[List[List[float]], List[List[int]]]]:
    """Load cached distance and time matrices if available and not expired."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        key = get_cache_key(coords, mode)
        if key in cache and cache[key]["coordinates"] == coords and cache[key]["mode"] == mode:
            timestamp = datetime.fromisoformat(cache[key]["timestamp"])
            if datetime.now() - timestamp < timedelta(days=CACHE_EXPIRY_DAYS):
                st.info("Matriz cargada desde caché.")
                return cache[key]["distance_matrix"], cache[key]["time_matrix"]
        return None
    except Exception as e:
        st.warning(f"Error leyendo caché: {e}")
        return None

def save_cached_matrices(
    coords: List[Tuple[float, float]],
    mode: str,
    dist_m: List[List[float]],
    time_m: List[List[int]]
) -> None:
    """Save distance and time matrices to cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        key = get_cache_key(coords, mode)
        cache[key] = {
            "coordinates": coords,
            "mode": mode,
            "distance_matrix": dist_m,
            "time_matrix": time_m,
            "timestamp": datetime.now().isoformat()
        }
        # Remove expired entries
        cache = {
            k: v for k, v in cache.items()
            if datetime.now() - datetime.fromisoformat(v["timestamp"]) < timedelta(days=CACHE_EXPIRY_DAYS)
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        st.info("Matriz guardada en caché.")
    except Exception as e:
        st.warning(f"Error guardando caché: {e}")