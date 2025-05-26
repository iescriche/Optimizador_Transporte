import requests
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import List, Tuple
from .utils import haversine
from .cache import load_cached_matrices, save_cached_matrices, get_cache_key
import pandas as pd

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def validate_ors_key(api_key: str) -> bool:
    """Valida la clave de ORS API con un request de prueba."""
    if not api_key:
        return False
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    body = {"coordinates": [[8.681495, 49.41461], [8.686507, 49.41943]]}
    try:
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        st.error(f"Error al validar clave ORS: {str(e)}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def ors_distance_matrix_block(
    origins: List[Tuple[float, float]],
    dests: List[Tuple[float, float]],
    api_key: str,
    mode: str = "driving-car",
) -> Tuple[List[List[float]], List[List[int]]]:
    """Obtiene matriz de distancias y tiempos desde ORS."""
    if not api_key:
        raise ValueError("Clave ORS no proporcionada.")
    for lat, lon in origins + dests:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or pd.isna(lat) or pd.isna(lon):
            raise ValueError(f"Coordenada inválida: ({lat}, {lon})")
    coords = [[lon, lat] for lat, lon in origins + dests]
    body = {
        "locations": coords,
        "sources": list(range(len(origins))),
        "destinations": list(range(len(origins), len(origins) + len(dests))),
        "metrics": ["distance", "duration"],
    }
    try:
        r = requests.post(
            f"https://api.openrouteservice.org/v2/matrix/{mode}",
            json=body,
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if "distances" not in data or "durations" not in data:
            raise ValueError("Respuesta de ORS no contiene matrices válidas.")
        dist = data["distances"]
        dur = [[int(x / 60 + 0.5) for x in row] for row in data["durations"]]
        if not (isinstance(dist, list) and isinstance(dur, list) and
                len(dist) == len(origins) and len(dur) == len(origins) and
                all(isinstance(row, list) and len(row) == len(dests) for row in dist) and
                all(isinstance(row, list) and len(row) == len(dests) for row in dur) and
                all(all(isinstance(x, (int, float)) and x >= 0 for x in row) for row in dist) and
                all(all(isinstance(x, int) and x >= 0 for x in row) for row in dur)):
            raise ValueError("Matrices de ORS tienen dimensiones o tipos inválidos.")
        return dist, dur
    except (requests.RequestException, ValueError, KeyError) as e:
        st.warning(f"Error en ORS matrix: {str(e)}. Usando Haversine como fallback.")
        raise ValueError(f"ORS API error: {str(e)}")
@st.cache_data(show_spinner=False, max_entries=10)
def ors_matrix_chunk(
    coords: List[Tuple[float, float]],
    api_key: str,
    block: int = 10,
    mode: str = "driving-car"
) -> Tuple[List[List[float]], List[List[int]]]:
    """Calcula matriz de distancias y tiempos por bloques, usando caché."""
    if not coords or len(coords) < 2:
        st.error("Lista de coordenadas vacía o insuficiente.")
        raise ValueError("Se requieren al menos dos coordenadas válidas.")
    
    # Validar coordenadas
    for lat, lon in coords:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or pd.isna(lat) or pd.isna(lon):
            st.error(f"Coordenada inválida: ({lat}, {lon}).")
            raise ValueError(f"Coordenada inválida: ({lat}, {lon})")

    # Intentar cargar desde caché
    cached = load_cached_matrices(coords, mode)
    if cached:
        dist_m, time_m = cached
        if (isinstance(dist_m, list) and isinstance(time_m, list) and
            len(dist_m) == len(coords) and len(time_m) == len(coords) and
            all(isinstance(row, list) and len(row) == len(coords) for row in dist_m) and
            all(isinstance(row, list) and len(row) == len(coords) for row in time_m)):
            return dist_m, time_m
        st.warning("Matriz en caché inválida. Recalculando.")
        logging.warning("Caché inválido para coords=%s", coords)

    # Calcular matriz
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    dur = [[0] * n for _ in range(n)]
    
    for r0 in range(0, n, block):
        for c0 in range(0, n, block):
            sub_orig = coords[r0 : r0 + block]
            sub_dest = coords[c0 : c0 + block]
            try:
                d_sub, t_sub = ors_distance_matrix_block(sub_orig, sub_dest, api_key, mode)
                # Validar submatrices
                if not (isinstance(d_sub, list) and isinstance(t_sub, list) and
                        len(d_sub) == len(sub_orig) and len(t_sub) == len(sub_orig) and
                        all(isinstance(row, list) and len(row) == len(sub_dest) for row in d_sub) and
                        all(isinstance(row, list) and len(row) == len(sub_dest) for row in t_sub)):
                    raise ValueError("Submatrices de ORS inválidas.")
            except ValueError as e:
                st.warning(f"Error con ORS API: {e}. Usando Haversine.")
                d_sub = [[haversine(*o, *d) for d in sub_dest] for o in sub_orig]
                speed_kmh = 30 if mode == "driving-car" else 15 if mode.startswith("cycling") else 5
                t_sub = [[int(d / 1000 / speed_kmh * 60 + 0.5) for d in row] for row in d_sub]
                if not (isinstance(d_sub, list) and isinstance(t_sub, list) and
                        len(d_sub) == len(sub_orig) and len(t_sub) == len(sub_orig) and
                        all(isinstance(row, list) and len(row) == len(sub_dest) for row in d_sub) and
                        all(isinstance(row, list) and len(row) == len(sub_dest) for row in t_sub)):
                    st.error("Fallo en el cálculo de Haversine.")
                    raise ValueError("Submatrices de Haversine inválidas.")
            
            for i, ri in enumerate(range(r0, min(r0 + block, n))):
                for j, cj in enumerate(range(c0, min(c0 + block, n))):
                    dist[ri][cj] = d_sub[i][j]
                    dur[ri][cj] = t_sub[i][j]
    
    # Rellenar simetría
    for i in range(n):
        for j in range(i + 1, n):
            dist[j][i] = dist[i][j]
            dur[j][i] = dur[i][j]

    # Validación final
    if not (isinstance(dist, list) and isinstance(dur, list) and
            len(dist) == n and len(dur) == n and
            all(isinstance(row, list) and len(row) == n for row in dist) and
            all(isinstance(row, list) and len(row) == n for row in dur)):
        st.error("Matrices finales inválidas.")
        raise ValueError("Las matrices finales no son válidas.")
    
    save_cached_matrices(coords, mode, dist, dur)
    return dist, dur

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_polyline_ors(
    orig: Tuple[float, float],
    dest: Tuple[float, float],
    api_key: str,
    mode: str = "driving-car"
) -> List[Tuple[float, float]]:
    """Obtiene polilínea de ruta desde ORS Directions API."""
    if not api_key:
        st.warning("Clave ORS no proporcionada. Usando línea recta como fallback.")
        return [(orig[0], orig[1]), (dest[0], dest[1])]
    url = f"https://api.openrouteservice.org/v2/directions/{mode}/geojson"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    body = {"coordinates": [[orig[1], orig[0]], [dest[1], dest[0]]], "geometry": True}
    try:
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "features" in data and data["features"]:
            return [(lat, lon) for lon, lat in data["features"][0]["geometry"]["coordinates"]]
        st.warning(f"No se encontraron rutas para {orig} -> {dest}")
        return []
    except requests.RequestException as e:
        st.warning(f"Error al conectar con ORS Directions API: {str(e)}. Usando línea recta.")
        return [(orig[0], orig[1]), (dest[0], dest[1])]