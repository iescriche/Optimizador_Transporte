import math
import unicodedata
import pandas as pd
import streamlit as st

def clean_addr(s: any) -> str:
    """Convierte a mayúsculas, elimina tildes y normaliza espacios, preservando comas.

    Args:
        s: Input string or value to clean.

    Returns:
        str: Cleaned address string.
    """
    if pd.isna(s) or s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.replace(",", " , ").upper().split())

def time_to_minutes(t: str) -> int:
    """Convierte HH:MM a minutos desde medianoche.

    Args:
        t: Time string in HH:MM format (24-hour).

    Returns:
        int: Minutes since midnight.

    Raises:
        ValueError: If format is invalid or time is out of range.
    """
    try:
        h, m = map(int, t.split(":"))
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError(f"Time out of range: {t} (use HH:MM, 00:00-23:59)")
        return h * 60 + m
    except ValueError as e:
        st.error(f"Formato de hora inválido: {t} (usa HH:MM)")
        raise ValueError(f"Invalid time format: {t} (use HH:MM)") from e

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia en metros usando la fórmula de Haversine.

    Args:
        lat1, lon1: Latitude and longitude of first point (degrees).
        lat2, lon2: Latitude and longitude of second point (degrees).

    Returns:
        float: Distance in meters.
    """
    EARTH_RADIUS = 6_371_000  # meters
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def validate_coords(lat: float, lon: float) -> bool:
    """Valida que las coordenadas estén en rangos válidos.

    Args:
        lat: Latitude (degrees).
        lon: Longitude (degrees).

    Returns:
        bool: True if coordinates are valid.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        st.warning(f"Coordenadas inválidas: lat={lat}, lon={lon}")
        return False
    return True