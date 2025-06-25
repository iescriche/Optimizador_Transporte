import sys
import os
import streamlit as st
import pandas as pd
import folium
import yaml
import json
import uuid
import io
import pickle
import hashlib
import logging
import traceback
from datetime import datetime
from streamlit_folium import st_folium
import altair as alt
import math
from typing import List, Tuple, Optional

# Configurar directorio raíz
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
logging.info("Directorio raíz agregado: %s", root_dir)

# Importar módulos del planificador
try:
    from route_planner import (
        load_data, ors_matrix_chunk, validate_ors_key, solve_vrp_simple, get_polyline_ors,
        recompute_etas, PlanningResult, Vehicle, reassign_nearby_stops
    )
except ImportError as e:
    logging.error("Error importando route_planner: %s", e)
    raise

# Configurar logging
logging.basicConfig(filename="route_planner.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MAX_H_REG = 8  # Regular hours limit
MAX_INT64 = 2**63 - 1  # Límite para int64_t
MAX_DAY_MINUTES = 1440  # Máximo minutos en un día

def time_to_minutes(time_val):
    """Convierte una cadena en formato HH:MM o un número a minutos."""
    if pd.isna(time_val):
        return None
    if isinstance(time_val, (int, float)):
        return int(time_val)
    try:
        hours, minutes = map(int, time_val.split(':'))
        if not (0 <= hours < 24 and 0 <= minutes < 60):
            raise ValueError(f"Hora fuera de rango: {time_val}")
        return hours * 60 + minutes
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Formato de tiempo inválido: {time_val}") from e

def validate_coords(lat: float, lon: float) -> bool:
    """Valida que las coordenadas estén en rangos válidos."""
    return -90 <= lat <= 90 and -180 <= lon <= 180

def generate_google_maps_link(lat: float, lng: float) -> str:
    """Genera un enlace de Google Maps para un destino."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"

def generate_google_maps_route(route: list, coords: list) -> str:
    """Genera un enlace de Google Maps para una ruta multi-parada."""
    waypoints = "/".join(f"{coords[node][0]},{coords[node][1]}" for node in route[1:-1])
    start = f"{coords[route[0]][0]},{coords[route[0]][1]}"
    end = f"{coords[route[-1]][0]},{coords[route[-1]][1]}"
    return f"https://www.google.com/maps/dir/{start}/{waypoints}/{end}"

def export_routes_to_excel(routes, addresses, eta, coords):
    """Exporta las rutas a un archivo Excel."""
    data = []
    for i, route in enumerate(routes):
        route_link = generate_google_maps_route(route, coords)
        for j, node in enumerate(route):
            eta_str = f"{eta[node] // 60:02d}:{eta[node] % 60:02d}" if node < len(eta) and eta[node] is not None else ""
            data.append({
                "Número de Vehículo": i + 1,
                "Orden de Parada": j,
                "Ubicación": addresses[node] if node < len(addresses) else "",
                "Hora Estimada de Llegada": eta_str,
                "Latitud": coords[node][0] if node < len(coords) else "",
                "Longitud": coords[node][1] if node < len(coords) else "",
                "Enlace Google Maps Parada": generate_google_maps_link(coords[node][0], coords[node][1]) if node < len(coords) else "",
                "Enlace Google Maps Ruta Completa": route_link if j == 0 else ""
            })
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer

def calculate_co2_emissions(routes, dist_m, fuel_consumption, fuel_type="diesel"):
    """Calcula emisiones de CO2 por vehículo en kg."""
    if fuel_consumption <= 0:
        st.warning("El consumo de combustible es 0 o inválido, las emisiones de CO2 serán 0.")
        logging.warning("Consumo de combustible inválido para CO2")
        return pd.DataFrame([{"Vehículo": f"Vehículo {i + 1}", "Emisiones de CO2 (kg)": 0.0} for i in range(len(routes))])
    
    emission_factors = {"diesel": 2.68, "gasoline": 2.31}
    emission_factor = emission_factors.get(fuel_type, 2.68)
    
    co2_emissions = []
    for i, route in enumerate(routes):
        if len(route) <= 2:
            co2_emissions.append({"Vehículo": f"Vehículo {i + 1}", "Emisiones de CO2 (kg)": 0.0})
        else:
            total_distance_km = sum(dist_m[route[j]][route[j + 1]] / 1000 for j in range(len(route) - 1))
            fuel_liters = total_distance_km * fuel_consumption / 100
            co2_kg = fuel_liters * emission_factor
            co2_emissions.append({"Vehículo": f"Vehículo {i + 1}", "Emisiones de CO2 (kg)": round(co2_kg, 2)})
    
    return pd.DataFrame(co2_emissions)

def reset_session_state():
    """Limpia las claves relevantes de st.session_state."""
    keys = ["plan", "kpi_df", "km_per_order", "euro_per_order", "co2_df", "map", "cost_optimizer_results"]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def get_cached_matrix(coords: List[Tuple[float, float]], api_key: str) -> Tuple[Optional[List[List[float]]], Optional[List[List[int]]]]:
    """Obtiene matrices de distancia/tiempo desde caché o API."""
    logging.info("Iniciando get_cached_matrix con %d coordenadas", len(coords))
    if not coords or len(coords) < 2:
        st.error("Coordenadas insuficientes.")
        logging.error("Coordenadas insuficientes: %s", coords)
        return None, None

    coords_hash = hashlib.md5(str(coords).encode()).hexdigest()
    cache_file = f"cache/matrix_{coords_hash}.pkl"
    os.makedirs("cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        try:
            dist_m, time_m = pickle.load(open(cache_file, "rb"))
            if (isinstance(dist_m, list) and isinstance(time_m, list) and
                len(dist_m) == len(coords) and len(time_m) == len(coords) and
                all(isinstance(row, list) and len(row) == len(coords) for row in dist_m) and
                all(isinstance(row, list) and len(row) == len(coords) for row in time_m)):
                logging.info("Usando matriz en caché: %s", cache_file)
                return dist_m, time_m
            st.warning("Caché corrupto en %s. Recalculando.", cache_file)
            os.remove(cache_file)
        except Exception as e:
            st.warning("Error al leer caché: %s. Recalculando.", e)
            os.remove(cache_file)

    logging.info("Calculando nueva matriz desde ORS API")
    try:
        dist_m, time_m = ors_matrix_chunk(coords, api_key)
        if not (isinstance(dist_m, list) and isinstance(time_m, list) and
                len(dist_m) == len(coords) and len(time_m) == len(coords) and
                all(isinstance(row, list) and len(row) == len(coords) for row in dist_m) and
                all(isinstance(row, list) and len(row) == len(coords) for row in time_m)):
            st.error("Matrices devueltas por ORS no son válidas.")
            logging.error("Matrices no válidas: dist_m=%s, time_m=%s", type(dist_m), type(time_m))
            raise ValueError("Matrices no válidas.")
        pickle.dump((dist_m, time_m), open(cache_file, "wb"))
        logging.info("Matriz guardada en caché: %s", cache_file)
        return dist_m, time_m
    except Exception as e:
        st.error("Error al calcular matrices: %s", e)
        logging.error("Error en ORS API: %s", e)
        return None, None

def validate_coordinates(df):
    """Valida que las columnas LATITUD y LONGITUD sean numéricas y estén en rangos válidos."""
    if not pd.api.types.is_numeric_dtype(df["LATITUD"]) or not pd.api.types.is_numeric_dtype(df["LONGITUD"]):
        st.error("Las columnas LATITUD y LONGITUD deben contener valores numéricos.")
        logging.error("Columnas LATITUD/LONGITUD no numéricas")
        return False
    if not (df["LATITUD"].between(-90, 90).all() and df["LONGITUD"].between(-180, 180).all()):
        st.error("Las coordenadas están fuera de los rangos válidos (LAT: -90 a 90, LON: -180 a 180).")
        logging.error("Coordenadas fuera de rango")
        return False
    if df["LATITUD"].isna().any() or df["LONGITUD"].isna().any():
        st.error("Las columnas LATITUD y LONGITUD contienen valores NaN.")
        logging.error("Coordenadas contienen NaN")
        return False
    return True

def validate_time_windows(df):
    """Valida que TIME_WINDOW_START y TIME_WINDOW_END sean válidos, convirtiendo HH:MM si es necesario."""
    if "TIME_WINDOW_START" not in df.columns or "TIME_WINDOW_END" not in df.columns:
        logging.info("No se proporcionaron columnas TIME_WINDOW_START/TIME_WINDOW_END")
        return True
    try:
        df["TIME_WINDOW_START"] = df["TIME_WINDOW_START"].apply(lambda x: time_to_minutes(x) if pd.notna(x) else 0)
        df["TIME_WINDOW_END"] = df["TIME_WINDOW_END"].apply(lambda x: time_to_minutes(x) if pd.notna(x) else MAX_DAY_MINUTES)
        # Clampear valores para asegurar que estén dentro de [0, 1440]
        df["TIME_WINDOW_START"] = df["TIME_WINDOW_START"].clip(lower=0, upper=MAX_DAY_MINUTES)
        df["TIME_WINDOW_END"] = df["TIME_WINDOW_END"].clip(lower=0, upper=MAX_DAY_MINUTES)
    except ValueError as e:
        st.error(f"Las columnas TIME_WINDOW_START y TIME_WINDOW_END contienen valores inválidos: {e}")
        logging.error("Error procesando TIME_WINDOW_START/TIME_WINDOW_END: %s", e)
        return False
    if df["TIME_WINDOW_START"].isna().any() or df["TIME_WINDOW_END"].isna().any():
        st.error("Las columnas TIME_WINDOW_START y TIME_WINDOW_END contienen valores NaN.")
        logging.error("TIME_WINDOW_START/TIME_WINDOW_END contienen NaN")
        return False
    if not (df["TIME_WINDOW_START"] <= df["TIME_WINDOW_END"]).all():
        st.error("TIME_WINDOW_START debe ser menor o igual que TIME_WINDOW_END.")
        logging.error("TIME_WINDOW_START mayor que TIME_WINDOW_END")
        return False
    if not (df["TIME_WINDOW_START"].between(0, MAX_DAY_MINUTES).all() and df["TIME_WINDOW_END"].between(0, MAX_DAY_MINUTES).all()):
        st.error(f"Las ventanas de tiempo deben estar entre 0 y {MAX_DAY_MINUTES} minutos.")
        logging.error(f"Ventanas de tiempo fuera de rango [0, {MAX_DAY_MINUTES}]")
        return False
    return True

def plot_kpi_bars(df: pd.DataFrame, column: str, title: str, y_label: str) -> alt.Chart:
    """Genera un gráfico de barras para un KPI específico."""
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Vehículo:N', title='Vehículo'),
        y=alt.Y(f'{column}:Q', title=y_label),
        color=alt.Color('Vehículo:N', legend=None)
    ).properties(
        title=title,
        width=400,
        height=300
    )
    return chart

def calculate_kpis(
    plan: List[List[int]],
    dist_m: List[List[float]],
    time_m: List[List[int]],
    df_today: pd.DataFrame,
    eta: List[Optional[int]],
    start_time_minutes: int,
    service_time: float = 0,
    price_per_hour: Optional[float] = None,
    fuel_price: Optional[float] = None,
    fuel_consumption: Optional[float] = None,
    optimize_for: str = "Coste",
    extra_max: Optional[float] = None,  # Máximo de horas extras permitidas
    extra_price: Optional[float] = None  # Tarifa por hora extra
) -> Tuple[pd.DataFrame, float, float]:
    """
    Calcula KPIs: km/ruta, km/pedido, euro/ruta, euro/pedido, incluyendo horas extras.

    Args:
        plan: Lista de rutas, donde cada ruta es una lista de índices de nodos.
        dist_m: Matriz de distancias en metros.
        time_m: Matriz de tiempos en minutos.
        df_today: DataFrame con datos de clientes (DIRECCION, LATITUD, LONGITUD, etc.).
        eta: Lista de tiempos estimados de llegada (minutos desde medianoche) o None.
        start_time_minutes: Hora de inicio en minutos desde medianoche.
        service_time: Tiempo de servicio por parada en minutos (default: 0).
        price_per_hour: Costo por hora del conductor en euros (opcional).
        fuel_price: Precio del combustible en euros por litro (opcional).
        fuel_consumption: Consumo de combustible en litros por 100 km (opcional).
        optimize_for: Criterio de optimización ("Coste" u otro, default: "Coste").
        extra_max: Máximo de horas extras permitidas por conductor (opcional, en horas).
        extra_price: Tarifa por hora extra en euros (opcional).

    Returns:
        Tuple: (DataFrame con KPIs, km por pedido, euros por pedido).

    Raises:
        ValueError: Si los parámetros de entrada son inválidos.
    """
    logging.info("Iniciando cálculo de KPIs")
    
    # Validar eta
    if eta is None or not isinstance(eta, list) or len(eta) != len(df_today):
        logging.error("Parámetro eta inválido: longitud=%s, esperado=%s", len(eta) if isinstance(eta, list) else 'None', len(df_today))
        st.error(f"Error: El parámetro eta es inválido o no coincide con el número de nodos ({len(df_today)}).")
        raise ValueError("El parámetro eta debe ser una lista con la misma longitud que df_today.")

    # Validar matrices
    n = len(df_today)
    if not (isinstance(dist_m, list) and isinstance(time_m, list) and
            len(dist_m) == n and len(time_m) == n and
            all(isinstance(row, list) and len(row) == n for row in dist_m) and
            all(isinstance(row, list) and len(row) == n for row in time_m)):
        st.error("Matriz de distancias o tiempos inválida.")
        logging.error("Matrices inválidas: dist_m=%s, time_m=%s", type(dist_m), type(time_m))
        raise ValueError("Matriz de distancias o tiempos no es una lista bidimensional válida.")

    # Validar parámetros de costos
    if price_per_hour is not None and price_per_hour < 0:
        logging.error("El precio por hora no puede ser negativo: %s", price_per_hour)
        raise ValueError("El precio por hora no puede ser negativo.")
    if extra_max is not None and extra_max < 0:
        logging.error("El máximo de horas extras no puede ser negativo: %s", extra_max)
        raise ValueError("El máximo de horas extras no puede ser negativo.")
    if extra_price is not None and extra_price < 0:
        logging.error("El precio de la hora extra no puede ser negativo: %s", extra_price)
        raise ValueError("El precio de la hora extra no puede ser negativo.")

    if not plan:
        st.warning("El plan de rutas está vacío.")
        logging.warning("Plan de rutas vacío")
        return pd.DataFrame(), 0, 0

    kpi_data = []
    total_distance = 0
    total_time = 0
    total_cost = 0
    total_stops = 0
    MAX_H_REG = 8  # Horas regulares por día (en horas)

    for v, route in enumerate(plan):
        if len(route) <= 2:
            logging.info(f"Ruta vacía o solo depósito para Vehículo {v + 1}, omitiendo")
            kpi_data.append({
                "Vehículo": f"Vehículo {v + 1}",
                "Kilómetros Recorridos": 0.0,
                "Tiempo Total (min)": 0.0,
                "Coste Total de la Ruta (€)": 0.0,
                "Paradas": 0,
                "Horas Extras (h)": 0.0
            })
            continue
        try:
            # Validar índices de la ruta
            if not all(isinstance(i, int) and 0 <= i < n for i in route):
                logging.error(f"Índices inválidos en ruta del Vehículo {v + 1}: {route}")
                st.error(f"Error: Ruta del Vehículo {v + 1} contiene índices inválidos.")
                continue

            # Calcular distancia y número de paradas
            route_distance = sum(dist_m[route[i]][route[i + 1]] / 1000 for i in range(len(route) - 1))
            stops = len(route) - 2
            if stops == 0:
                logging.info(f"Sin paradas válidas para Vehículo {v + 1}, omitiendo")
                kpi_data.append({
                    "Vehículo": f"Vehículo {v + 1}",
                    "Kilómetros Recorridos": 0.0,
                    "Tiempo Total (min)": 0.0,
                    "Coste Total de la Ruta (€)": 0.0,
                    "Paradas": 0,
                    "Horas Extras (h)": 0.0
                })
                continue

            # Calcular tiempo total
            last_stop_idx = route[-2] if len(route) > 2 else route[-1]
            if last_stop_idx >= len(eta):
                logging.error(f"Índice de última parada inválido para Vehículo {v + 1}: {last_stop_idx}, eta len={len(eta)}")
                st.error(f"Error: Índice de última parada inválido para Vehículo {v + 1}.")
                continue

            if eta[last_stop_idx] is not None:
                route_time = eta[last_stop_idx] - start_time_minutes
            else:
                # Fallback: Calcular tiempo basado en time_m y service_time
                logging.warning(f"ETA None para última parada del Vehículo {v + 1}, usando time_m")
                route_time = sum(time_m[route[i]][route[i + 1]] for i in range(len(route) - 1))
                route_time += stops * service_time
            route_time = max(route_time, 0)  # Evitar tiempos negativos

            # Convertir tiempo a horas para el cálculo de costos
            route_time_hours = route_time / 60  # Convertir minutos a horas

            # Calcular horas regulares y extras
            regular_hours = min(route_time_hours, MAX_H_REG)
            extra_hours = max(0, route_time_hours - MAX_H_REG)
            if extra_max is not None:
                extra_hours = min(extra_hours, extra_max)  # Limitar horas extras

            # Calcular costo total
            route_cost = 0
            if optimize_for == "Coste" and fuel_price is not None and fuel_consumption is not None and fuel_price > 0 and fuel_consumption > 0:
                fuel_cost = (route_distance / 100) * fuel_consumption * fuel_price
                route_cost += fuel_cost
                logging.debug(f"Vehículo {v + 1}: Costo de combustible = {fuel_cost:.2f}€")
            if optimize_for == "Coste" and price_per_hour is not None and price_per_hour > 0:
                driver_cost = regular_hours * price_per_hour
                if extra_price is not None and extra_hours > 0:
                    driver_cost += extra_hours * extra_price
                else:
                    driver_cost += extra_hours * price_per_hour * 1.5  # Tarifa por defecto para horas extras (50% más)
                route_cost += driver_cost
                logging.debug(f"Vehículo {v + 1}: Horas regulares = {regular_hours:.2f}, Horas extras = {extra_hours:.2f}, Costo laboral = {driver_cost:.2f}€")

            kpi_data.append({
                "Vehículo": f"Vehículo {v + 1}",
                "Kilómetros Recorridos": round(route_distance, 2),
                "Tiempo Total (min)": round(route_time, 2),
                "Coste Total de la Ruta (€)": round(route_cost, 2) if route_cost > 0 and optimize_for == "Coste" else 0,
                "Paradas": stops,
                "Horas Extras (h)": round(extra_hours, 2)
            })

            total_distance += route_distance
            total_time += route_time
            total_cost += route_cost if route_cost > 0 and optimize_for == "Coste" else 0
            total_stops += stops
            logging.info(f"KPIs calculados para Vehículo {v + 1}: Distancia={route_distance:.2f}km, Tiempo={route_time:.2f}min, Coste={route_cost:.2f}€, Horas extras={extra_hours:.2f}h")
        except IndexError as e:
            st.error(f"Error al calcular KPIs para Vehículo {v + 1}: Índice fuera de rango ({e})")
            logging.error(f"IndexError en Vehículo {v + 1}: {e}")
            continue
        except Exception as e:
            st.error(f"Error inesperado al calcular KPIs para Vehículo {v + 1}: {e}")
            logging.error(f"Error inesperado en Vehículo {v + 1}: {e}", exc_info=True)
            continue

    kpi_data.append({
        "Vehículo": "Total",
        "Kilómetros Recorridos": round(total_distance, 2),
        "Tiempo Total (min)": round(total_time, 2),
        "Coste Total de la Ruta (€)": round(total_cost, 2) if total_cost > 0 and optimize_for == "Coste" else 0,
        "Paradas": total_stops,
        "Horas Extras (h)": round(sum(item["Horas Extras (h)"] for item in kpi_data), 2)
    })

    kpi_df = pd.DataFrame(kpi_data)
    km_per_order = total_distance / total_stops if total_stops > 0 else 0
    time_per_order = total_time / total_stops if total_stops > 0 else 0
    euro_per_order = total_cost / total_stops if total_stops > 0 and total_cost > 0 and optimize_for == "Coste" else 0
    logging.info(f"KPIs globales: km_per_order={km_per_order:.2f}, time_per_order={time_per_order:.2f}min, euro_per_order={euro_per_order:.2f}")

    return kpi_df, km_per_order, euro_per_order

def main():
    st.set_page_config(
        page_title="💰 Cost Optimizer",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("💰 Cost Optimizer")
    st.markdown("Optimiza rutas para minimizar costos, considerando salarios, horas extra y combustible.")

    # Initialize session state
    for key in ["plan", "kpi_df", "km_per_order", "euro_per_order", "co2_df", "map", "cost_optimizer_results"]:
        if key not in st.session_state:
            st.session_state[key] = None

    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo config.yaml.")
        logging.error("Archivo config.yaml no encontrado")
        return
    except yaml.YAMLError as e:
        st.error(f"Error en el formato de config.yaml: {e}")
        logging.error(f"Error en config.yaml: %s", e)
        return

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.markdown("---")

        with st.expander("📂 Carga de Datos", expanded=True):
            st.markdown("**Selecciona la fuente de datos**")
            mode = st.selectbox("Fuente de datos", ["Automática", "Subir archivo CSV", "Subir archivos Excel"], key="cost_data_source")
            df_today = pd.DataFrame()
            if mode == "Automática":
                try:
                    df_today = load_data("clientes.xlsx", "ruta.xlsx", config)
                except Exception as e:
                    st.error(f"Error cargando datos automáticos: {e}")
                    logging.error(f"Error en carga automática: %s", e)
                    return
            elif mode == "Subir archivo CSV":
                up_file = st.file_uploader("Subir archivo CSV", type=["csv"], key="cost_csv_upload")
                if not up_file:
                    st.info("Sube un archivo CSV para continuar.")
                    return
                try:
                    df_today = pd.read_csv(up_file)
                    st.write("Vista previa de datos:")
                    st.dataframe(df_today.head())
                    if not all(col in df_today.columns for col in ["DIRECCION", "LATITUD", "LONGITUD"]):
                        st.error("El archivo CSV debe contener las columnas 'DIRECCION', 'LATITUD', 'LONGITUD'.")
                        logging.error("Faltan columnas en CSV")
                        return
                except Exception as e:
                    st.error(f"Error leyendo archivo CSV: {e}")
                    logging.error(f"Error leyendo CSV: %s", e)
                    return
            else:
                up_cli = st.file_uploader("Maestro clientes (xlsx/csv)", type=["xlsx", "xls", "csv"], key="cost_clients_upload")
                up_rta = st.file_uploader("Rutas del día (xlsx/csv)", type=["xlsx", "xls", "csv"], key="cost_routes_upload")
                if not up_cli or not up_rta:
                    st.info("Sube ambos ficheros para continuar.")
                    return
                try:
                    clients_df = pd.read_excel(up_cli) if up_cli.name.endswith((".xlsx", ".xls")) else pd.read_csv(up_cli)
                    routes_df = pd.read_excel(up_rta) if up_rta.name.endswith((".xlsx", ".xls")) else pd.read_csv(up_rta)
                    st.write("Vista previa de clientes:")
                    st.dataframe(clients_df.head())
                    st.write("Vista previa de rutas:")
                    st.dataframe(routes_df.head())
                except Exception as e:
                    st.error(f"Error leyendo ficheros: {e}")
                    logging.error(f"Error leyendo ficheros Excel/CSV: %s", e)
                    return

                st.subheader("Mapear Columnas")
                default_cols = config.get("default_columns", {})
                col_address = st.selectbox(
                    "Columna de Dirección",
                    clients_df.columns,
                    index=clients_df.columns.tolist().index(default_cols.get("address", "DIRECCION")) if default_cols.get("address", "DIRECCION") in clients_df.columns else 0,
                    key="cost_col_address"
                )
                col_lat = st.selectbox(
                    "Columna de Latitud",
                    clients_df.columns,
                    index=clients_df.columns.tolist().index(default_cols.get("latitude", "LATITUD")) if default_cols.get("latitude", "LATITUD") in clients_df.columns else 0,
                    key="cost_col_lat"
                )
                col_lon = st.selectbox(
                    "Columna de Longitud",
                    clients_df.columns,
                    index=clients_df.columns.tolist().index(default_cols.get("longitude", "LONGITUD")) if default_cols.get("longitude", "LONGITUD") in clients_df.columns else 0,
                    key="cost_col_lon"
                )
                col_route = st.selectbox(
                    "Columna de Ruta",
                    routes_df.columns,
                    index=routes_df.columns.tolist().index("RUTA") if "RUTA" in routes_df.columns else 0,
                    key="cost_col_route"
                )
                col_time_start = st.selectbox(
                    "Columna de Inicio de Ventana de Tiempo",
                    routes_df.columns,
                    index=routes_df.columns.tolist().index("HORA_INI") if "HORA_INI" in routes_df.columns else 0,
                    key="cost_col_time_start"
                )
                col_time_end = st.selectbox(
                    "Columna de Fin de Ventana de Tiempo",
                    routes_df.columns,
                    index=routes_df.columns.tolist().index("HORA_FIN") if "HORA_FIN" in routes_df.columns else 0,
                    key="cost_col_time_end"
                )
                
                column_mapping = {
                    "address": col_address,
                    "latitude": col_lat,
                    "longitude": col_lon,
                    "route": col_route,
                    "time_window_start": col_time_start,
                    "time_window_end": col_time_end
                }
                if st.button("💾 Guardar Mapeo", key="cost_save_mapping"):
                    with open("column_mapping.json", "w") as f:
                        json.dump(column_mapping, f, indent=2)
                    st.success("Mapeo de columnas guardado en column_mapping.json")
                    logging.info("Mapeo de columnas guardado")
                
                try:
                    df_today = load_data(clients_df, routes_df, config, column_mapping)
                except Exception as e:
                    st.error(f"Error procesando datos: {e}")
                    logging.error(f"Error procesando datos: %s", e)
                    return
            
            if df_today.empty:
                st.warning("No se cargaron datos válidos.")
                logging.warning("df_today vacío")
                return
            
            # Validar coordenadas
            if not validate_coordinates(df_today):
                invalid_coords = df_today[~df_today.apply(lambda row: validate_coords(row["LATITUD"], row["LONGITUD"]), axis=1)]
                if not invalid_coords.empty:
                    st.error("Coordenadas inválidas encontradas:")
                    st.dataframe(invalid_coords[["DIRECCION", "LATITUD", "LONGITUD"]])
                return
            
            # Validar y convertir ventanas de tiempo
            if not validate_time_windows(df_today):
                st.error("Por favor, corrija los datos de TIME_WINDOW_START y TIME_WINDOW_END.")
                return
            
            # Selección manual del depósito
            st.subheader("Seleccionar Depósito")
            depot_address = st.selectbox(
                "Dirección del almacén",
                df_today["DIRECCION"].unique(),
                key="cost_depot_address",
                help="Elige la dirección que será el punto de partida y llegada de las rutas."
            )
            depot_idx = df_today[df_today["DIRECCION"] == depot_address].index
            if len(depot_idx) != 1:
                st.error("No se pudo identificar un índice único para el depósito. Verifica las direcciones.")
                logging.error("Múltiples o ningún índice para el depósito: %s", depot_address)
                return
            depot_idx = depot_idx[0]
            logging.info("Depósito seleccionado: %s, índice original: %d", depot_address, depot_idx)
            
            # Reordenar df_today para que el depósito esté en el índice 0
            df_today = pd.concat([df_today.loc[[depot_idx]], df_today.drop(depot_idx)]).reset_index(drop=True)
            depot_idx = 0
            logging.info("Depósito reordenado al índice: %d, dirección: %s", depot_idx, df_today.at[depot_idx, 'DIRECCION'])

        with st.expander("🛠️ Configuración de Rutas"):
            respect_predefined = st.checkbox("Respetar rutas predefinidas", value=False, key="cost_respect_predefined_routes")
            min_vehs = 1
            vehs = None
            rutas_predef = 0
            predefined_routes = None
            if respect_predefined and "RUTA" in df_today.columns:
                route_groups = df_today.groupby("RUTA")
                predefined_routes = []
                for route_name, group in route_groups:
                    if route_name and route_name.strip():
                        indices = group.index.tolist()
                        if depot_idx in indices:
                            indices.remove(depot_idx)
                        if indices:
                            predefined_routes.append(indices)
                if not predefined_routes:
                    st.error("No se encontraron rutas predefinidas válidas en la columna RUTA.")
                    return
                rutas_predef = len(predefined_routes)
                st.info(f"Se detectaron **{rutas_predef}** rutas predefinidas; se usará ese número de furgonetas.")
                vehs = rutas_predef
                min_vehs = vehs
            else:
                vehs = st.slider("Furgonetas máximas disponibles", min_value=1, max_value=20, value=5, key="cost_vehicles")
                min_vehs = 1

            balance = st.checkbox("Balancear rutas", value=True, key="cost_balance_routes")
            reassign_stops = False
            reassign_distance = 5000.0
            if respect_predefined:
                reassign_stops = st.checkbox("Permitir reasignar paradas cercanas", key="cost_reassign_stops")
                if reassign_stops:
                    reassign_distance = st.number_input(
                        "Distancia máxima para reasignar paradas (km)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, key="cost_reassign_distance"
                    ) * 1000

            start_time = st.time_input("Hora de salida", value=datetime.strptime(config["default_settings"]["start_time"], "%H:%M").time(), key="cost_start_time")
            start_time_minutes = int(start_time.hour * 60 + start_time.minute)
            if not (0 <= start_time_minutes <= MAX_DAY_MINUTES):
                st.error("La hora de salida debe estar entre 00:00 y 23:59.")
                logging.error("start_time_minutes fuera de rango: %d", start_time_minutes)
                return
            service_time = st.number_input("Tiempo de servicio por parada (min)", min_value=0, max_value=60, value=config["default_settings"]["service_time"], key="cost_service_time")
            balance_threshold = st.slider("Umbral de balanceo (%)", 50, 100, 90, key="cost_balance_threshold") / 100
            max_stops_per_vehicle = st.number_input(
                "Máximo de paradas por vehículo (0 para sin límite)",
                min_value=0,
                value=0,
                help="Define el número máximo de paradas por vehículo. Usa 0 para no limitar.",
                key="cost_max_stops"
            )
            relax_time_windows = st.checkbox("Relajar ventanas de tiempo", value=True, key="cost_relax_time_windows")
            time_window_buffer = 0
            if relax_time_windows:
                time_window_buffer = st.number_input(
                    "Margen adicional para ventanas de tiempo (minutos)",
                    min_value=0,
                    max_value=240,
                    value=0,
                    step=15,
                    key="cost_time_window_buffer"
                )
            recomendar = st.checkbox("Recomendar número óptimo de vehículos", value=True, key="cost_recomendar")
            solver_timeout = st.number_input(
                "Tiempo máximo del solver (segundos)",
                min_value=30,
                max_value=300,
                value=120,
                step=10,
                key="cost_solver_timeout"
            )

        with st.expander("💶 Costes"):
            price_per_hour = st.number_input(
                "Sueldo por hora del repartidor (€/hora)", min_value=0.0, value=10.0, step=0.1, key="cost_price_per_hour"
            )
            extra_max = st.number_input(
                "Máx. horas extra por conductor", min_value=0.0, max_value=12.0, step=0.5, value=2.0, key="cost_extra_max_h"
            )
            extra_price = st.number_input(
                "Precio de la hora extra (€/h)", min_value=0.0, step=0.1, value=price_per_hour * 1.5, key="cost_extra_price_h"
            )
            max_minutes = int(st.number_input(
                "Tiempo máximo por vehículo (horas)",
                min_value=1.0,
                max_value=24.0,
                value=24.0,
                step=0.5,
                key="cost_max_minutes"
            ) * 60)
            if max_minutes > MAX_INT64:
                st.error(f"Tiempo máximo por vehículo excede el límite de int64_t ({MAX_INT64} minutos).")
                logging.error("max_minutes excede int64_t: %d", max_minutes)
                return

            fuel_type = st.selectbox("Tipo de combustible", ["Diésel", "Gasolina"], key="cost_fuel_type")
            fuel_type_key = 'diesel' if fuel_type == "Diésel" else 'gasoline'
            default_prices = {'diesel': 1.376, 'gasoline': 1.467}
            fuel_price = st.number_input(
                "Precio del combustible (€/litro)", min_value=0.0, value=default_prices[fuel_type_key], step=0.01, key="cost_fuel_price"
            )
            fuel_consumption = st.number_input(
                "Consumo de combustible (litros/100 km)", min_value=0.0, value=8.0, step=0.1, key="cost_fuel_consumption"
            )
            st.info(f"Precio del {fuel_type.lower()} por defecto: {default_prices[fuel_type_key]:.3f} €/litro")

        with st.expander("🔑 API"):
            api_key = st.text_input("ORS API-Key", value=config.get("ors_api_key", os.getenv("ORS_API_KEY", "")), type="password", key="cost_api_key")
            if st.button("Probar clave ORS", key="cost_test_ors_key"):
                if validate_ors_key(api_key):
                    st.success("Clave ORS válida.")
                    logging.info("Clave ORS válida")
                else:
                    st.error("Clave ORS inválida.")
                    logging.error("Clave ORS inválida")

        st.markdown("---")
        if st.button("🚚 Calcular Rutas", key="cost_calculate_routes", use_container_width=True):
            reset_session_state()
            if not api_key or not validate_ors_key(api_key):
                st.error("Clave ORS inválida o no proporcionada.")
                logging.error("Clave ORS inválida o no proporcionada")
                return

            if df_today.empty or not all(col in df_today.columns for col in ["DIRECCION", "LATITUD", "LONGITUD"]):
                st.error("Datos inválidos. Asegúrate de que el archivo contiene las columnas 'DIRECCION', 'LATITUD', 'LONGITUD'.")
                logging.error("Datos inválidos o faltan columnas")
                return

            try:
                with st.spinner("Calculando rutas óptimas..."):
                    logging.info("Iniciando cálculo de rutas")
                    coords = list(zip(df_today["LATITUD"], df_today["LONGITUD"]))
                    dist_m_global, time_m_global = get_cached_matrix(coords, api_key)
                    if dist_m_global is None or time_m_global is None:
                        st.error("No se pudieron obtener las matrices de distancia y tiempo.")
                        logging.error("Matrices inválidas: dist_m=%s, time_m=%s", dist_m_global, time_m_global)
                        return
                    
                    # Validar número de nodos
                    if len(df_today) != len(coords):
                        st.error(f"El número de filas en df_today ({len(df_today)}) no coincide con el número de coordenadas ({len(coords)}).")
                        logging.error("Mismatch: len(df_today)=%d, len(coords)=%d", len(df_today), len(coords))
                        return
                    if len(dist_m_global) != len(df_today):
                        st.error(f"El tamaño de la matriz de distancias ({len(dist_m_global)}) no coincide con el número de nodos ({len(df_today)}).")
                        logging.error("Mismatch: len(dist_m_global)=%d, len(df_today)=%d", len(dist_m_global), len(df_today))
                        return

                    # Extraer ventanas de tiempo con relajación
                    time_windows = None
                    if "TIME_WINDOW_START" in df_today.columns and "TIME_WINDOW_END" in df_today.columns:
                        time_windows = []
                        invalid_windows = []
                        for idx, (start, end) in enumerate(zip(df_today["TIME_WINDOW_START"], df_today["TIME_WINDOW_END"])):
                            try:
                                start_val = int(float(start) + time_window_buffer) if pd.notna(start) else 0
                                end_val = int(float(end) + time_window_buffer) if pd.notna(end) else MAX_DAY_MINUTES
                                # Clampear valores para no exceder MAX_DAY_MINUTES
                                start_val = min(max(0, start_val), MAX_DAY_MINUTES)
                                end_val = min(max(start_val, end_val), MAX_DAY_MINUTES)
                                time_windows.append((start_val, end_val))
                                logging.debug("Ventana de tiempo para nodo %d: [%d, %d]", idx, start_val, end_val)
                            except (TypeError, ValueError) as e:
                                invalid_windows.append((idx, start, end, str(e)))
                                time_windows.append((0, MAX_DAY_MINUTES))
                        if invalid_windows:
                            st.warning("Se detectaron ventanas de tiempo inválidas. Usando [0, 1440] para los nodos afectados:")
                            for idx, start, end, error in invalid_windows:
                                st.write(f"- Nodo {idx}: start={start}, end={end}, error={error}")
                            logging.warning("Ventanas de tiempo inválidas en %d nodos: %s", len(invalid_windows), invalid_windows)
                    else:
                        time_windows = [(0, MAX_DAY_MINUTES)] * len(df_today)
                        logging.info("Ventanas de tiempo no proporcionadas, usando [0, %d] para todos los nodos", MAX_DAY_MINUTES)

                    n_stops = len(coords) - 1
                    fixed_hours = 2
                    fixed_cents = int(round(fixed_hours * price_per_hour * 100))

                    resultados = []
                    if respect_predefined and predefined_routes:
                        routes, eta, used = solve_vrp_simple(
                            dist_m=dist_m_global,
                            time_m=time_m_global,
                            vehicles=vehs,
                            depot_idx=depot_idx,
                            balance=balance,
                            start_min=start_time_minutes,
                            service_time=service_time,
                            time_windows=time_windows,
                            max_stops_per_vehicle=len(coords) if max_stops_per_vehicle == 0 else max_stops_per_vehicle,
                            balance_threshold=balance_threshold,
                            predefined_routes=predefined_routes,
                            respect_predefined=True,
                            fuel_price=fuel_price,
                            fuel_consumption=fuel_consumption,
                            price_per_hour=price_per_hour,
                            max_minutes=max_minutes,
                            cost_mode=True
                        )
                        if not routes:
                            st.error("No se encontraron rutas válidas para las rutas predefinidas. Intenta relajar restricciones.")
                            return
                        if reassign_stops:
                            routes, eta, used = reassign_nearby_stops(
                                routes=routes,
                                dist_m=dist_m_global,
                                time_m=time_m_global,
                                depot_idx=depot_idx,
                                balance=balance,
                                start_min=start_time_minutes,
                                service_time=service_time,
                                time_windows=time_windows,
                                max_stops_per_vehicle=len(coords) if max_stops_per_vehicle == 0 else max_stops_per_vehicle,
                                max_distance_m=reassign_distance,
                                balance_threshold=balance_threshold,
                                fuel_price=fuel_price,
                                fuel_consumption=fuel_consumption,
                                price_per_hour=price_per_hour,
                                max_minutes=max_minutes
                            )
                        eta = recompute_etas(routes, time_m_global, start_time_minutes, service_time, len(df_today), time_windows, depot_idx)
                        if eta is None or not isinstance(eta, list) or len(eta) != len(df_today):
                            st.error("Error: ETA no está definido correctamente.")
                            logging.error("ETA inválido: %s", eta)
                            return
                        kpi_df, km_per_order, euro_per_order = calculate_kpis(
                            plan=routes,
                            dist_m=dist_m_global,
                            time_m=time_m_global,
                            df_today=df_today,
                            eta=eta,
                            start_time_minutes=start_time_minutes,
                            service_time=service_time,
                            price_per_hour=price_per_hour if price_per_hour > 0 else None,
                            fuel_price=fuel_price if fuel_price > 0 else None,
                            fuel_consumption=fuel_consumption if fuel_consumption > 0 else None,
                            optimize_for="Coste",
                            extra_max=extra_max,
                            extra_price=extra_price
                        )
                        coste_variable = kpi_df["Coste Total de la Ruta (€)"].sum() if not kpi_df.empty and kpi_df["Coste Total de la Ruta (€)"].notna().any() else 0
                        coste_fijo = vehs * fixed_cents / 100
                        coste_total = coste_fijo + coste_variable
                        resultados.append((vehs, coste_total, routes, kpi_df, eta, km_per_order, euro_per_order))
                    else:
                        for v in range(min_vehs, vehs + 1):
                            if recomendar:
                                recommended_v = max(min_vehs, math.ceil(n_stops / (math.ceil(n_stops / v) + 2)))
                                if v < recommended_v:
                                    continue
                            plan, eta, used = solve_vrp_simple(
                                dist_m=dist_m_global,
                                time_m=time_m_global,
                                vehicles=v,
                                depot_idx=depot_idx,
                                balance=balance,
                                start_min=start_time_minutes,
                                service_time=service_time,
                                time_windows=time_windows,
                                max_stops_per_vehicle=len(coords) if max_stops_per_vehicle == 0 else max_stops_per_vehicle,
                                balance_threshold=balance_threshold,
                                fuel_price=fuel_price,
                                fuel_consumption=fuel_consumption,
                                price_per_hour=price_per_hour,
                                max_minutes=max_minutes,
                                cost_mode=True
                            )
                            if not plan:
                                st.warning(f"No se encontró solución para {v} vehículos. Probando con más vehículos...")
                                continue
                            eta = recompute_etas(plan, time_m_global, start_time_minutes, service_time, len(df_today), time_windows, depot_idx)
                            if eta is None or not isinstance(eta, list) or len(eta) != len(df_today):
                                st.error("Error: ETA no está definido correctamente.")
                                logging.error("ETA inválido: %s", eta)
                                return
                            kpi_df, km_per_order, euro_per_order = calculate_kpis(
                                plan=plan,
                                dist_m=dist_m_global,
                                time_m=time_m_global,
                                df_today=df_today,
                                eta=eta,
                                start_time_minutes=start_time_minutes,
                                service_time=service_time,
                                price_per_hour=price_per_hour if price_per_hour > 0 else None,
                                fuel_price=fuel_price if fuel_price > 0 else None,
                                fuel_consumption=fuel_consumption if fuel_consumption > 0 else None,
                                optimize_for="Coste",
                                extra_max=extra_max,
                                extra_price=extra_price
                            )
                            coste_variable = kpi_df["Coste Total de la Ruta (€)"].sum() if not kpi_df.empty and kpi_df["Coste Total de la Ruta (€)"].notna().any() else 0
                            coste_fijo = v * fixed_cents / 100
                            coste_total = coste_fijo + coste_variable
                            resultados.append((v, coste_total, plan, kpi_df, eta, km_per_order, euro_per_order))

                    if not resultados:
                        st.error("No se encontraron rutas válidas para ningún número de vehículos. Intenta relajar restricciones (más vehículos, mayor tiempo máximo, o sin ventanas de tiempo).")
                        return

                    opt_v, opt_coste, opt_plan, opt_kpi, eta_global, km_per_order, euro_per_order = min(resultados, key=lambda t: t[1])
                    assigned_vehicles = len([rt for rt in opt_plan if len(rt) > 2])
                    st.success(f"Nº óptimo de furgonetas: {opt_v} — Coste total: {opt_coste:,.2f} €")

                    # Cost curve chart
                    df_curve = pd.DataFrame(
                        [(v, c) for v, c, *_ in resultados],
                        columns=["Vehículos", "Coste_total €"]
                    )
                    cost_chart = alt.Chart(df_curve).mark_line(point=True).encode(
                        x="Vehículos:O",
                        y="Coste_total €:Q"
                    ).properties(
                        width=400,
                        height=250,
                        title="Coste total vs número de furgonetas"
                    )
                    st.altair_chart(cost_chart, use_container_width=True)

                    # Create map
                    fmap = folium.Map(location=df_today[["LATITUD", "LONGITUD"]].mean().tolist(), zoom_start=10, tiles="OpenStreetMap")
                    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] * 5
                    polyline_failures = 0
                    links = []
                    for v, rt in enumerate(opt_plan):
                        color = palette[v % len(palette)]
                        for i in range(len(rt) - 1):
                            try:
                                orig = (df_today.at[rt[i], "LATITUD"], df_today.at[rt[i], "LONGITUD"])
                                dest = (df_today.at[rt[i + 1], "LATITUD"], df_today.at[rt[i + 1], "LONGITUD"])
                                pts = get_polyline_ors(orig, dest, api_key)
                                if pts:
                                    folium.PolyLine(pts, color=color, weight=4).add_to(fmap)
                                else:
                                    polyline_failures += 1
                                    folium.PolyLine([orig, dest], color=color, weight=4, dash_array="5, 5").add_to(fmap)
                                    logging.warning(f"Fallo en polilínea para segmento {orig} -> {dest}")
                            except Exception as e:
                                logging.warning("Error al dibujar segmento para vehículo %d: %s", v + 1, e)
                                polyline_failures += 1
                        for seq, node in enumerate(rt):
                            try:
                                eta_str = f"{eta_global[node] // 60:02d}:{eta_global[node] % 60:02d}" if eta_global[node] else "N/A"
                                popup_html = f"""
                                V{v + 1}·{seq} {df_today.at[node, 'DIRECCION']}<br>
                                Hora Estimada de Llegada: {eta_str}<br>
                                <a href="{generate_google_maps_link(df_today.at[node, 'LATITUD'], df_today.at[node, 'LONGITUD'])}" target="_blank">Abrir en Google Maps</a>
                                """
                                folium.CircleMarker(
                                    location=(df_today.at[node, 'LATITUD'], df_today.at[node, 'LONGITUD']),
                                    radius=6 if seq == 0 else 4,
                                    color=color,
                                    fill=True,
                                    popup=folium.Popup(popup_html, max_width=300)
                                ).add_to(fmap)
                            except Exception as e:
                                logging.warning("Error al dibujar marcador para vehículo %d, nodo %d: %s", v + 1, node, e)
                        wps = [f"{df_today.at[n, 'LATITUD']},{df_today.at[n, 'LONGITUD']}" for n in rt[1:-1]]
                        url = f"https://www.google.com/maps/dir/{coords[rt[0]][0]},{coords[rt[0]][1]}/{'/'.join(wps)}/{coords[rt[0]][0]},{coords[rt[0]][1]}"
                        links.append({"Vehículo": v + 1, "Link": url})

                    if polyline_failures > 0:
                        st.warning(f"Se usaron líneas rectas para {polyline_failures} segmentos debido a errores en la API de ORS.")
                        logging.warning(f"{polyline_failures} fallos en polilíneas")

                    # Create plan
                    plan = PlanningResult(
                        generated_at=datetime.now().isoformat(),
                        depot=depot_idx,
                        coordinates=coords,
                        addresses=df_today["DIRECCION"].tolist(),
                        routes=[Vehicle(v + 1, r, [False] * len(r)) for v, r in enumerate(opt_plan)],
                        distance_matrix_hash=str(hash(str(coords))),
                        settings={
                            "service_time": service_time,
                            "start_time": start_time_minutes,
                            "vehicles": opt_v,
                            "balance": balance,
                            "balance_threshold": balance_threshold,
                            "fuel_price": fuel_price,
                            "fuel_consumption": fuel_consumption,
                            "price_per_hour": price_per_hour,
                            "fuel_type": fuel_type_key,
                            "api_key": api_key,
                            "respect_predefined": respect_predefined,
                            "max_distance_m": reassign_distance
                        },
                        distance_matrix=dist_m_global,
                        time_matrix=time_m_global,
                        eta=eta_global
                    )

                    # Save plan
                    plan_file = f"plans/{datetime.now().strftime('%Y-%m-%d')}_cost_plan_{uuid.uuid4().hex[:8]}.json"
                    os.makedirs("plans", exist_ok=True)
                    with open(plan_file, "w") as f:
                        json.dump(plan.asdict(), f, indent=2)
                    logging.info(f"Plan guardado en {plan_file}")

                    # Calculate CO2 emissions
                    co2_df = calculate_co2_emissions(opt_plan, dist_m_global, fuel_consumption, fuel_type_key)

                    # Store in session state
                    st.session_state["kpi_df"] = opt_kpi
                    st.session_state["km_per_order"] = km_per_order
                    st.session_state["euro_per_order"] = euro_per_order
                    st.session_state["co2_df"] = co2_df
                    st.session_state["plan"] = plan
                    st.session_state["map"] = fmap
                    st.session_state["cost_optimizer_results"] = {
                        "opt_v": opt_v,
                        "opt_coste": opt_coste,
                        "links": pd.DataFrame(links)
                    }
                    st.success("Rutas calculadas con éxito!")
                    logging.info("Rutas calculadas con éxito")

            except Exception as e:
                st.error(f"Error calculando rutas: {e}")
                logging.error(f"Error calculando rutas: %s\n%s", e, "".join(traceback.format_stack()[:-1]))
                st.session_state["plan"] = None
                return

    if "plan" in st.session_state and st.session_state["plan"] is not None and st.session_state["plan"].routes:
        st.header("📈 Resultados")
        st.markdown("---")
        
        with st.container(border=True):
            st.subheader("Indicadores Clave")
            col1, col2, col3, col4, col5 = st.columns(5)
            total_stops = sum(len(r.sequence) - 2 for r in st.session_state["plan"].routes if len(r.sequence) > 2)
            col1.metric("Paradas Asignadas", total_stops)
            col2.metric("Furgonetas Utilizadas", st.session_state["cost_optimizer_results"]["opt_v"])
            col3.metric("Kilómetros por Pedido", f"{st.session_state['km_per_order']:.2f} km")
            col4.metric("Coste por Pedido", f"{st.session_state['euro_per_order']:.2f} €" if st.session_state['euro_per_order'] else "N/A")
            col5.metric("Coste Total", f"{st.session_state['cost_optimizer_results']['opt_coste']:.2f} €")

        with st.container(border=True):
            st.subheader("Análisis por Vehículo")
            if not st.session_state["kpi_df"].empty:
                st.dataframe(
                    st.session_state["kpi_df"],
                    use_container_width=True,
                    column_config={
                        "Vehículo": st.column_config.TextColumn("Vehículo"),
                        "Kilómetros Recorridos": st.column_config.NumberColumn("Kilómetros (km)", format="%.2f"),
                        "Tiempo Total (min)": st.column_config.NumberColumn("Tiempo (min)", format="%.2f"),
                        "Coste Total de la Ruta (€)": st.column_config.NumberColumn("Coste (€)", format="%.2f"),
                        "Paradas": st.column_config.NumberColumn("Paradas"),
                        "Horas Extras (h)": st.column_config.NumberColumn("Horas Extras (h)", format="%.2f", help="Horas extras realizadas por el conductor")
                    }
                )
                excel_kpi_buffer = io.BytesIO()
                st.session_state["kpi_df"].to_excel(excel_kpi_buffer, index=False, engine="openpyxl")
                st.download_button(
                    label="📥 Descargar Indicadores de Rendimiento (Excel)",
                    data=excel_kpi_buffer,
                    file_name=f"kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="cost_download_kpis_excel"
                )

                col_chart1, col_chart2, col_chart3 = st.columns(3)
                with col_chart1:
                    km_chart = plot_kpi_bars(st.session_state["kpi_df"], "Kilómetros Recorridos", "Kilómetros por Ruta", "Kilómetros")
                    st.altair_chart(km_chart, use_container_width=True)
                with col_chart2:
                    if st.session_state["kpi_df"]["Coste Total de la Ruta (€)"].notna().any():
                        euro_chart = plot_kpi_bars(st.session_state["kpi_df"], "Coste Total de la Ruta (€)", "Coste por Ruta", "Euros")
                        st.altair_chart(euro_chart, use_container_width=True)
                    else:
                        st.warning("Coste por ruta no calculado: introduce sueldo por hora y/o consumo de combustible.")
                with col_chart3:
                    if st.session_state["kpi_df"]["Tiempo Total (min)"].notna().any():
                        time_chart = plot_kpi_bars(st.session_state["kpi_df"], "Tiempo Total (min)", "Tiempo por Ruta", "Minutos")
                        st.altair_chart(time_chart, use_container_width=True)
                    else:
                        st.warning("Tiempo no calculado: verifica los datos de entrada.")

        with st.container(border=True):
            st.subheader("Itinerario de Vehículos")
            vehicles_with_routes = [v for v in st.session_state["plan"].routes if len(v.sequence) > 2]
            if vehicles_with_routes:
                tabs = st.tabs([f"Vehículo {v.vehicle_id}" for v in vehicles_with_routes])
                rows = []
                for tab, vehicle in zip(tabs, vehicles_with_routes):
                    with tab:
                        st.markdown(f"**Ruta Completa de Vehículo {vehicle.vehicle_id}**")
                        route_link = generate_google_maps_route(vehicle.sequence, st.session_state["plan"].coordinates)
                        st.markdown(f"[Abrir en Google Maps]({route_link})")
                        for seq, node in enumerate(vehicle.sequence[1:-1], 1):
                            eta_val = st.session_state["plan"].eta[node] if node < len(st.session_state["plan"].eta) else None
                            eta_str = f"{eta_val // 60:02d}:{eta_val % 60:02d}" if eta_val else "N/A"
                            rows.append({
                                "Número de Vehículo": vehicle.vehicle_id,
                                "Orden de Parada": seq,
                                "Ubicación": st.session_state["plan"].addresses[node],
                                "Hora Estimada de Llegada": eta_str
                            })
                        tab.dataframe(
                            pd.DataFrame(rows[-len(vehicle.sequence[1:-1]):]),
                            use_container_width=True,
                            column_config={
                                "Número de Vehículo": st.column_config.NumberColumn("Vehículo"),
                                "Orden de Parada": st.column_config.NumberColumn("Orden"),
                                "Ubicación": st.column_config.TextColumn("Ubicación"),
                                "Hora Estimada de Llegada": st.column_config.TextColumn(
                                    "Hora Estimada",
                                    help="Hora estimada de llegada en formato HH:MM"
                                )
                            }
                        )
                
                if rows:
                    df_routes = pd.DataFrame(rows)
                    st.download_button(
                        label="📥 Descargar Itinerario (Excel)",
                        data=export_routes_to_excel(
                            [v.sequence for v in st.session_state["plan"].routes],
                            st.session_state["plan"].addresses,
                            st.session_state["plan"].eta,
                            st.session_state["plan"].coordinates
                        ),
                        file_name=f"planificacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="cost_download_plan_excel"
                    )
            else:
                st.info("No hay itinerarios de vehículos para mostrar.")

        with st.container(border=True):
            st.subheader("🔗 Links para Conductores")
            if "cost_optimizer_results" in st.session_state and st.session_state["cost_optimizer_results"]["links"] is not None:
                st.dataframe(
                    st.session_state["cost_optimizer_results"]["links"],
                    use_container_width=True,
                    column_config={
                        "Link": st.column_config.LinkColumn("Link", display_text="Abrir Ruta")
                    }
                )
            else:
                st.info("No hay links para conductores disponibles.")

        with st.container(border=True):
            st.subheader("🗺️ Mapa de Rutas")
            if st.session_state["map"]:
                st_folium(st.session_state["map"], use_container_width=True, height=600, key="cost_route_map")
                html_buffer = io.BytesIO()
                st.session_state["map"].save(html_buffer, close_file=False)
                html_buffer.seek(0)
                st.download_button(
                    label="📥 Descargar Mapa (HTML)",
                    data=html_buffer,
                    file_name=f"mapa_rutas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    key="cost_download_map"
                )
            else:
                st.info("No hay mapa disponible para mostrar.")

if __name__ == "__main__":
    main()