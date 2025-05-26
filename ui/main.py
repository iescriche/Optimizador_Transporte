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
from datetime import datetime
from streamlit_folium import st_folium
from typing import List, Tuple, Optional

# Configurar directorio raíz
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
print(f"Directorio raíz agregado: {root_dir}")
print(f"sys.path: {sys.path}")

# Importar módulos del planificador
try:
    from route_planner import (
        load_data, ors_matrix_chunk, validate_ors_key, solve_vrp_simple, get_polyline_ors,
        calculate_kpis, recompute_etas, PlanningResult, Vehicle, reassign_nearby_stops
    )
except ImportError as e:
    print(f"Error importando route_planner: {e}")
    raise

# Configurar logging
logging.basicConfig(filename="route_planner.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuración de la página
st.set_page_config(
    page_title="🚚 Standard Route Planner",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            continue
        total_distance_km = sum(dist_m[route[j]][route[j + 1]] / 1000 for j in range(len(route) - 1))
        fuel_liters = total_distance_km * fuel_consumption / 100
        co2_kg = fuel_liters * emission_factor
        co2_emissions.append({"Vehículo": f"Vehículo {i + 1}", "Emisiones de CO2 (kg)": round(co2_kg, 2)})
    
    return pd.DataFrame(co2_emissions)

def reset_session_state():
    """Limpia las claves relevantes de st.session_state."""
    keys = ["plan", "kpi_df", "km_per_order", "euro_per_order", "co2_df", "map", "links"]
    for key in keys:
        st.session_state[key] = None

def get_cached_matrix(coords: List[Tuple[float, float]], api_key: str) -> Tuple[Optional[List[List[float]]], Optional[List[List[int]]]]:
    """Obtiene matrices de distancia/tiempo desde caché o API."""
    print("Iniciando get_cached_matrix...")
    print(f"Número de coordenadas: {len(coords)}")
    if not coords or len(coords) < 2:
        st.error("Coordenadas insuficientes.")
        logging.error("Coordenadas insuficientes: %s", coords)
        print("Error: Coordenadas insuficientes.")
        return None, None

    coords_hash = hashlib.md5(str(coords).encode()).hexdigest()
    cache_file = f"cache/matrix_{coords_hash}.pkl"
    os.makedirs("cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        try:
            dist_m, time_m = pickle.load(open(cache_file, "rb"))
            print(f"Caché encontrado: {cache_file}")
            if (isinstance(dist_m, list) and isinstance(time_m, list) and
                len(dist_m) == len(coords) and len(time_m) == len(coords) and
                all(isinstance(row, list) and len(row) == len(coords) for row in dist_m) and
                all(isinstance(row, list) and len(row) == len(coords) for row in time_m)):
                logging.info(f"Usando matriz en caché: {cache_file}")
                print("Caché válido, devolviendo matrices.")
                return dist_m, time_m
            st.warning(f"Caché corrupto en {cache_file}. Recalculando.")
            os.remove(cache_file)
            print(f"Caché corrupto en {cache_file}, recalculando.")
        except Exception as e:
            st.warning(f"Error al leer caché: {e}. Recalculando.")
            os.remove(cache_file)
            print(f"Error al leer caché: {e}")

    logging.info("Calculando nueva matriz desde ORS API")
    print("Calculando nueva matriz desde ORS API...")
    try:
        dist_m, time_m = ors_matrix_chunk(coords, api_key)
        print(f"Matriz de distancia: tipo={type(dist_m)}, tamaño={len(dist_m) if isinstance(dist_m, list) else 'N/A'}")
        print(f"Matriz de tiempo: tipo={type(time_m)}, tamaño={len(time_m) if isinstance(time_m, list) else 'N/A'}")
        if not (isinstance(dist_m, list) and isinstance(time_m, list) and
                len(dist_m) == len(coords) and len(time_m) == len(coords) and
                all(isinstance(row, list) and len(row) == len(coords) for row in dist_m) and
                all(isinstance(row, list) and len(row) == len(coords) for row in time_m)):
            st.error("Matrices devueltas por ORS no son válidas.")
            logging.error("Matrices no válidas: dist_m=%s, time_m=%s", type(dist_m), type(time_m))
            print("Error: Matrices no válidas devueltas por ORS.")
            raise ValueError("Matrices no válidas.")
        pickle.dump((dist_m, time_m), open(cache_file, "wb"))
        logging.info(f"Matriz guardada en caché: {cache_file}")
        print(f"Matriz guardada en caché: {cache_file}")
        return dist_m, time_m
    except Exception as e:
        st.error(f"Error al calcular matrices: {e}")
        logging.error("Error en ORS API: %s", e)
        print(f"Error al calcular matrices: {e}")
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

def standard_planner():
    """Interfaz principal del planificador de rutas."""
    st.markdown(
        """
        <style>
        .main { background-color: #f5f7fa; padding: 20px; border-radius: 10px; }
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
        .stButton>button:hover { background-color: #45a049; }
        .metric-card { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sidebar .sidebar-content { background-color: #e8ecef; }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🚚 Standard Route Planner")
    st.markdown("**Optimiza tus rutas de entrega con OpenRouteService**", unsafe_allow_html=True)
    st.markdown("Planifica rutas eficientes para tus vehículos, minimizando tiempo, distancia y costos.")

    # Inicializar estado de sesión
    for key in ["plan", "kpi_df", "km_per_order", "euro_per_order", "co2_df", "map", "links"]:
        if key not in st.session_state:
            st.session_state[key] = None

    logging.info("Iniciando Standard Route Planner")
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo config.yaml.")
        logging.error("Archivo config.yaml no encontrado")
        return
    except yaml.YAMLError as e:
        st.error(f"Error en el formato de config.yaml: {e}")
        logging.error(f"Error en config.yaml: {e}")
        return

    # Validar claves requeridas en config
    required_keys = ["default_settings"]
    for key in required_keys:
        if key not in config:
            st.error(f"Falta la clave '{key}' en config.yaml.")
            logging.error(f"Falta clave en config.yaml: {key}")
            return

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Configuración", anchor=False)
        st.markdown("---")

        with st.expander("📂 Carga de Datos", expanded=True):
            st.markdown("**Selecciona la fuente de datos**")
            mode = st.selectbox(
                "Fuente de datos",
                ["Automática", "Subir archivo CSV", "Subir archivos Excel"],
                key="data_source",
                help="Elige cómo cargar los datos de clientes y rutas."
            )
            df_today = pd.DataFrame()
            if mode == "Automática":
                try:
                    df_today = load_data("clientes.xlsx", "ruta.xlsx", config)
                except Exception as e:
                    st.error(f"Error cargando datos automáticos: {e}")
                    logging.error(f"Error en carga automática: {e}")
                    return
            elif mode == "Subir archivo CSV":
                up_file = st.file_uploader("Subir archivo CSV", type=["csv"], key="csv_upload")
                if not up_file:
                    st.info("Sube un archivo CSV para continuar.")
                    return
                try:
                    df_today = pd.read_csv(up_file)
                    if not all(col in df_today.columns for col in ["DIRECCION", "LATITUD", "LONGITUD"]):
                        st.error("El archivo CSV debe contener las columnas 'DIRECCION', 'LATITUD', 'LONGITUD'.")
                        logging.error("Faltan columnas en CSV")
                        return
                except Exception as e:
                    st.error(f"Error leyendo archivo CSV: {e}")
                    logging.error(f"Error leyendo CSV: {e}")
                    return
            else:
                up_cli = st.file_uploader("Maestro clientes (xlsx/csv)", type=["xlsx", "xls", "csv"], key="clients_upload")
                up_rta = st.file_uploader("Rutas del día (xlsx/csv)", type=["xlsx", "xls", "csv"], key="routes_upload")
                if not up_cli or not up_rta:
                    st.info("Sube ambos ficheros para continuar.")
                    return
                
                try:
                    clients_df = pd.read_excel(up_cli) if up_cli.name.endswith((".xlsx", ".xls")) else pd.read_csv(up_cli)
                    routes_df = pd.read_excel(up_rta) if up_rta.name.endswith((".xlsx", ".xls")) else pd.read_csv(up_rta)
                except Exception as e:
                    st.error(f"Error leyendo ficheros: {e}")
                    logging.error(f"Error leyendo ficheros Excel/CSV: {e}")
                    return
                
                st.subheader("Mapear Columnas", anchor=False)
                default_cols = config.get("default_columns", {})
                col_address = st.selectbox(
                    "Columna de Dirección",
                    clients_df.columns,
                    index=clients_df.columns.tolist().index(default_cols.get("address", "DIRECCION")) if default_cols.get("address", "DIRECCION") in clients_df.columns else 0,
                    key="col_address"
                )
                col_lat = st.selectbox(
                    "Columna de Latitud",
                    clients_df.columns,
                    index=clients_df.columns.tolist().index(default_cols.get("latitude", "LATITUD")) if default_cols.get("latitude", "LATITUD") in clients_df.columns else 0,
                    key="col_lat"
                )
                col_lon = st.selectbox(
                    "Columna de Longitud",
                    clients_df.columns,
                    index=clients_df.columns.tolist().index(default_cols.get("longitude", "LONGITUD")) if default_cols.get("longitude", "LONGITUD") in clients_df.columns else 0,
                    key="col_lon"
                )
                col_route = st.selectbox(
                    "Columna de Ruta",
                    routes_df.columns,
                    index=routes_df.columns.tolist().index("RUTA") if "RUTA" in routes_df.columns else 0,
                    key="col_route"
                )
                col_time_start = st.selectbox(
                    "Columna de Inicio de Ventana de Tiempo",
                    routes_df.columns,
                    index=routes_df.columns.tolist().index("HORA_INI") if "HORA_INI" in routes_df.columns else 0,
                    key="col_time_start"
                )
                col_time_end = st.selectbox(
                    "Columna de Fin de Ventana de Tiempo",
                    routes_df.columns,
                    index=routes_df.columns.tolist().index("HORA_FIN") if "HORA_FIN" in routes_df.columns else 0,
                    key="col_time_end"
                )
                
                column_mapping = {
                    "address": col_address,
                    "latitude": col_lat,
                    "longitude": col_lon,
                    "route": col_route,
                    "time_window_start": col_time_start,
                    "time_window_end": col_time_end
                }
                if st.button("💾 Guardar Mapeo", key="save_mapping"):
                    with open("column_mapping.json", "w") as f:
                        json.dump(column_mapping, f, indent=2)
                    st.success("Mapeo de columnas guardado en column_mapping.json")
                    logging.info("Mapeo de columnas guardado")
                
                try:
                    df_today = load_data(clients_df, routes_df, config, column_mapping)
                except Exception as e:
                    st.error(f"Error procesando datos: {e}")
                    logging.error(f"Error procesando datos: {e}")
                    return
            
            if df_today.empty:
                st.warning("No se cargaron datos válidos.")
                logging.warning("df_today vacío")
                return
            
            if not validate_coordinates(df_today):
                return
            
            # Selección manual del depósito
            st.subheader("Seleccionar Depósito", anchor=False)
            depot_address = st.selectbox(
                "Dirección del almacén",
                df_today["DIRECCION"].unique(),
                key="depot_address",
                help="Elige la dirección que será el punto de partida y llegada de las rutas."
            )
            depot_idx = df_today[df_today["DIRECCION"] == depot_address].index
            if len(depot_idx) != 1:
                st.error("No se pudo identificar un índice único para el depósito. Verifica las direcciones.")
                logging.error("Múltiples o ningún índice para el depósito: %s", depot_address)
                return
            depot_idx = depot_idx[0]
            print(f"Depósito seleccionado: {depot_address}, índice original: {depot_idx}")
            
            # Reordenar df_today para que el depósito esté en el índice 0
            df_today = pd.concat([df_today.loc[[depot_idx]], df_today.drop(depot_idx)]).reset_index(drop=True)
            depot_idx = 0  # Después de reordenar, el depósito siempre está en el índice 0
            print(f"Depósito reordenado al índice: {depot_idx}, dirección: {df_today.at[depot_idx, 'DIRECCION']}")

        with st.expander("🛠️ Configuración de Rutas"):
            respect_predefined = st.checkbox("Respetar rutas predefinidas", value=False, key="respect_predefined_routes")
            vehs = None
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
                for route in predefined_routes:
                    if not all(0 <= idx < len(df_today) for idx in route):
                        st.error(f"Ruta predefinida inválida con índices fuera de rango: {route}")
                        logging.error("Ruta predefinida inválida: %s", route)
                        return
                st.info(f"Se detectaron **{len(predefined_routes)}** rutas predefinidas; se usará ese número de furgonetas.")
                vehs = len(predefined_routes)
                min_vehs = vehs
            else:
                vehs = st.number_input("🚛 Furgonetas disponibles", min_value=1, max_value=60, value=config["default_settings"]["vehicles"], key="vehicles")
                min_vehs = 1

            balance = st.checkbox("⚖️ Balancear rutas", value=config["default_settings"]["balance"], key="balance_routes")
            reassign_stops = False
            reassign_distance = 5000.0
            if respect_predefined:
                reassign_stops = st.checkbox("🔄 Reasignar paradas cercanas", key="reassign_stops")
                if reassign_stops:
                    reassign_distance = st.number_input(
                        "📏 Distancia máxima para reasignar (km)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, key="reassign_distance"
                    ) * 1000

            start_time = st.time_input("🕒 Hora de salida", value=datetime.strptime(config["default_settings"]["start_time"], "%H:%M").time(), key="start_time")
            start_time_minutes = start_time.hour * 60 + start_time.minute
            service_time = st.number_input("⏱️ Tiempo de servicio por parada (min)", min_value=0, value=config["default_settings"]["service_time"], key="service_time")
            balance_threshold = None
            if balance:
                balance_threshold = st.number_input(
                    "⚖️ Umbral de balanceo (0 a 1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=config["default_settings"].get("balance_threshold", 0.9),
                    step=0.1,
                    help="Define la tolerancia para equilibrar la carga entre vehículos (0 = sin balanceo, 1 = balanceo perfecto).",
                    key="balance_threshold"
                )
            max_stops_per_vehicle = st.number_input(
                "📍 Máximo de paradas por vehículo (0 para sin límite)",
                min_value=0,
                value=0,
                help="Define el número máximo de paradas por vehículo. Usa 0 para no limitar.",
                key="max_stops"
            )

        with st.expander("💶 Costes"):
            fuel_price = st.number_input("⛽ Precio del combustible (€/litro)", min_value=0.0, value=config["default_settings"]["fuel_price"], key="fuel_price")
            fuel_consumption = st.number_input("🚗 Consumo de combustible (litros/100 km)", min_value=0.0, value=config["default_settings"]["fuel_consumption"], key="fuel_consumption")
            price_per_hour = st.number_input("💸 Sueldo por hora (€/hora)", min_value=0.0, value=config["default_settings"]["price_per_hour"], key="price_per_hour")

        with st.expander("🔑 API"):
            api_key = st.text_input("🔐 ORS API-Key", value=config.get("ors_api_key", os.getenv("ORS_API_KEY", "")), type="password", key="api_key")
            if st.button("🔍 Probar clave ORS", key="test_ors_key"):
                if validate_ors_key(api_key):
                    st.success("Clave ORS válida.")
                    logging.info("Clave ORS válida")
                else:
                    st.error("Clave ORS inválida.")
                    logging.error("Clave ORS inválida")

        st.markdown("---")
        if st.button("🚚 Calcular Rutas", key="calculate_routes", use_container_width=True):
            reset_session_state()
            if not api_key or not validate_ors_key(api_key):
                st.error("Clave ORS inválida o no proporcionada.")
                logging.error("Clave ORS inválida o no proporcionada")
                print("Error: Clave ORS inválida o no proporcionada.")
                return
            
            if df_today.empty or not all(col in df_today.columns for col in ["DIRECCION", "LATITUD", "LONGITUD"]):
                st.error("Datos inválidos. Asegúrate de que el archivo contiene las columnas 'DIRECCION', 'LATITUD', 'LONGITUD'.")
                logging.error("Datos inválidos o faltan columnas")
                print("Error: Datos inválidos o faltan columnas.")
                return
            
            try:
                with st.spinner("Calculando rutas, por favor espera..."):
                    logging.info("Iniciando cálculo de rutas")
                    print("Iniciando cálculo de rutas...")
                    coords = list(zip(df_today["LATITUD"], df_today["LONGITUD"]))
                    print(f"Coordenadas: {coords}")
                    dist_m, time_m = get_cached_matrix(coords, api_key)
                    print(f"Después de get_cached_matrix: dist_m tipo={type(dist_m)}, time_m tipo={type(time_m)}")
                    if dist_m is None or time_m is None:
                        st.error("No se pudieron obtener las matrices de distancia y tiempo.")
                        logging.error("Matrices inválidas: dist_m=%s, time_m=%s", dist_m, time_m)
                        print("Error: No se pudieron obtener las matrices.")
                        return
                    max_stops_per_vehicle = None if max_stops_per_vehicle == 0 else max_stops_per_vehicle
                    
                    # Validar número de nodos
                    if len(df_today) != len(coords):
                        st.error(f"El número de filas en df_today ({len(df_today)}) no coincide con el número de coordenadas ({len(coords)}).")
                        logging.error("Mismatch: len(df_today)=%d, len(coords)=%d", len(df_today), len(coords))
                        return
                    if len(dist_m) != len(df_today):
                        st.error(f"El tamaño de la matriz de distancias ({len(dist_m)}) no coincide con el número de nodos ({len(df_today)}).")
                        logging.error("Mismatch: len(dist_m)=%d, len(df_today)=%d", len(dist_m), len(df_today))
                        return

                    # Extraer ventanas de tiempo
                    time_windows = None
                    if "TIME_WINDOW_START" in df_today.columns and "TIME_WINDOW_END" in df_today.columns:
                        time_windows = list(zip(df_today["TIME_WINDOW_START"], df_today["TIME_WINDOW_END"]))
                        print(f"Ventanas de tiempo: {time_windows}")
                    else:
                        time_windows = [(0, 1440)] * len(df_today)  # Default: sin restricciones
                        print("No se proporcionaron ventanas de tiempo, usando [0, 1440] para todos los nodos.")

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
                            print("Error: No se encontraron rutas predefinidas válidas.")
                            return
                        print(f"Rutas predefinidas: {predefined_routes}")
                    
                    print("Llamando a solve_vrp_simple...")
                    print(f"Parámetros: n={len(df_today)}, vehicles={vehs}, depot={depot_idx}")
                    routes, eta, used = solve_vrp_simple(
                        dist_m=dist_m,
                        time_m=time_m,
                        vehicles=vehs,
                        depot=depot_idx,
                        balance=balance,
                        start_min=start_time_minutes,
                        service_time=service_time,
                        time_windows=time_windows,
                        max_stops_per_vehicle=max_stops_per_vehicle,
                        balance_threshold=balance_threshold,
                        predefined_routes=predefined_routes,
                        respect_predefined=respect_predefined
                    )
                    print(f"Resultado de solve_vrp_simple: routes={routes}, eta={eta}, used={used}")
                    
                    if not all(isinstance(route, list) for route in routes):
                        st.error("Rutas inválidas: algunos elementos no son listas.")
                        logging.error("Rutas inválidas: %s", routes)
                        print(f"Error: Rutas inválidas: {routes}")
                        return
                    
                    if respect_predefined and reassign_stops and predefined_routes:
                        print("Reasignando paradas cercanas...")
                        routes, eta, used = reassign_nearby_stops(
                            routes=routes,
                            dist_m=dist_m,
                            time_m=time_m,
                            depot=depot_idx,
                            balance=balance,
                            start_min=start_time_minutes,
                            service_time=service_time,
                            max_stops_per_vehicle=max_stops_per_vehicle,
                            max_distance_m=reassign_distance,
                            balance_threshold=balance_threshold,
                            time_windows=time_windows
                        )
                        print(f"Resultado de reassign_nearby_stops: routes={routes}, eta={eta}, used={used}")
                    
                    print("Recalculando ETAs...")
                    eta = recompute_etas(
                        routes=routes,
                        time_m=time_m,
                        start_min=start_time_minutes,
                        service_time=service_time,
                        n=len(df_today),
                        time_windows=time_windows
                    )
                    print(f"ETAs recalculados: {eta}")
                    
                    print("Creando mapa...")
                    fmap = folium.Map(location=df_today[["LATITUD", "LONGITUD"]].mean().tolist(), zoom_start=10)
                    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] * 5
                    polyline_failures = 0
                    links = []
                    
                    for v, route in enumerate(routes):
                        if not isinstance(route, list):
                            st.warning(f"Ruta inválida para vehículo {v + 1}: {route}. Omitiendo.")
                            print(f"Ruta inválida para vehículo {v + 1}: {route}")
                            continue
                        color = palette[v % len(palette)]
                        for i in range(len(route) - 1):
                            try:
                                orig = (df_today.at[route[i], "LATITUD"], df_today.at[route[i], "LONGITUD"])
                                dest = (df_today.at[route[i + 1], "LATITUD"], df_today.at[route[i + 1], "LONGITUD"])
                                pts = get_polyline_ors(orig, dest, api_key)
                                if pts:
                                    folium.PolyLine(pts, color=color, weight=4).add_to(fmap)
                                else:
                                    polyline_failures += 1
                                    folium.PolyLine([orig, dest], color=color, weight=4, dash_array="5, 5").add_to(fmap)
                                    logging.warning(f"Fallo en polilínea para segmento {orig} -> {dest}")
                            except Exception as e:
                                st.warning(f"Error al dibujar segmento para vehículo {v + 1}: {e}")
                                print(f"Error al dibujar segmento para vehículo {v + 1}: {e}")
                                polyline_failures += 1
                        
                        for seq, node in enumerate(route):
                            try:
                                eta_str = f"{eta[node] // 60:02d}:{eta[node] % 60:02d}" if eta[node] is not None else "N/A"
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
                                st.warning(f"Error al dibujar marcador para vehículo {v + 1}, nodo {node}: {e}")
                                print(f"Error al dibujar marcador para vehículo {v + 1}, nodo {node}: {e}")
                        
                        wps = [f"{df_today.at[n, 'LATITUD']},{df_today.at[n, 'LONGITUD']}" for n in route[1:-1]]
                        url = f"https://www.google.com/maps/dir/{coords[route[0]][0]},{coords[route[0]][1]}/{'/'.join(wps)}/{coords[route[0]][0]},{coords[route[0]][1]}"
                        links.append({"Vehículo": v, "Link": url})

                    if polyline_failures > 0:
                        st.warning(f"Se usaron líneas rectas para {polyline_failures} segmentos debido a errores en la API de ORS.")
                        logging.warning(f"{polyline_failures} fallos en polilíneas")
                        print(f"{polyline_failures} fallos en polilíneas")
                    
                    print("Creando objeto PlanningResult...")
                    plan = PlanningResult(
                        generated_at=datetime.now().isoformat(),
                        depot=depot_idx,
                        coordinates=coords,
                        addresses=df_today["DIRECCION"].tolist(),
                        routes=[Vehicle(v + 1, r, [False] * len(r)) for v, r in enumerate(routes)],
                        distance_matrix_hash=str(hash(str(coords))),
                        settings={
                            "service_time": service_time,
                            "start_time": start_time_minutes,
                            "vehicles": vehs,
                            "balance": balance,
                            "balance_threshold": balance_threshold,
                            "fuel_price": fuel_price,
                            "fuel_consumption": fuel_consumption,
                            "price_per_hour": price_per_hour,
                            "fuel_type": config["default_settings"]["fuel_type"],
                            "api_key": api_key,
                            "respect_predefined": respect_predefined,
                            "max_distance_m": reassign_distance
                        },
                        distance_matrix=dist_m,
                        time_matrix=time_m,
                        eta=eta
                    )
                    
                    print("Guardando plan...")
                    plan_file = f"plans/{datetime.now().strftime('%Y-%m-%d')}_plan_{uuid.uuid4().hex[:8]}.json"
                    os.makedirs("plans", exist_ok=True)
                    with open(plan_file, "w") as f:
                        json.dump(plan.asdict(), f, indent=2)
                    logging.info(f"Plan guardado en {plan_file}")
                    print(f"Plan guardado en {plan_file}")
                    
                    print("Calculando KPIs...")
                    kpi_df, km_per_order, euro_per_order = calculate_kpis(
                        routes, dist_m, time_m, df_today, eta, start_time_minutes, service_time,
                        price_per_hour, fuel_price, fuel_consumption
                    )
                    print("KPIs calculados.")
                    
                    print("Calculando emisiones de CO2...")
                    co2_df = calculate_co2_emissions(routes, dist_m, fuel_consumption, config["default_settings"]["fuel_type"])
                    print("Emisiones de CO2 calculadas.")
                    
                    st.session_state["kpi_df"] = kpi_df
                    st.session_state["km_per_order"] = km_per_order
                    st.session_state["euro_per_order"] = euro_per_order
                    st.session_state["co2_df"] = co2_df
                    st.session_state["plan"] = plan
                    st.session_state["map"] = fmap
                    st.session_state["links"] = pd.DataFrame(links)
                    st.success("✅ Rutas calculadas con éxito!")
                    logging.info("Rutas calculadas con éxito")
                    print("Rutas calculadas con éxito!")
            except Exception as e:
                st.error(f"❌ Error calculando rutas: {e}")
                logging.error(f"Error calculando rutas: {e}")
                print(f"Error calculando rutas: {e}")
                st.session_state["plan"] = None
                return

    if "plan" in st.session_state and st.session_state["plan"] is not None and st.session_state["plan"].routes:
        st.header("📊 Resultados", anchor=False)
        st.divider()

        # Indicadores Clave
        st.subheader("📈 Indicadores Clave", anchor=False)
        total_distance_single_km = 0
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            total_stops = sum(len(r.sequence) - 2 for r in st.session_state["plan"].routes if len(r.sequence) > 2)
            with col1:
                st.metric(
                    label="📍 Paradas Asignadas",
                    value=total_stops,
                    help="Total de paradas asignadas a los vehículos."
                )
            with col2:
                st.metric(
                    label="🚛 Furgonetas Utilizadas",
                    value=len([r for r in st.session_state["plan"].routes if len(r.sequence) > 2]),
                    help="Número de vehículos utilizados en las rutas."
                )
            with col3:
                st.metric(
                    label="🛣️ Kilómetros por Pedido",
                    value=f"{st.session_state['km_per_order']:.2f} km",
                    help="Distancia promedio por pedido."
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric(
                    label="💶 Coste por Pedido",
                    value=f"{st.session_state['euro_per_order']:.2f} €" if st.session_state['euro_per_order'] else "N/A",
                    help="Costo promedio por pedido."
                )
            with col5:
                total_co2_optimized = st.session_state["co2_df"]["Emisiones de CO2 (kg)"].sum() if not st.session_state["co2_df"].empty else 0
                st.metric(
                    label="🌱 CO2 Optimizado",
                    value=f"{total_co2_optimized:.2f} kg",
                    help="Emisiones totales de CO2 para las rutas optimizadas."
                )
            with col6:
                dist_m = st.session_state["plan"].distance_matrix
                n = len(st.session_state["plan"].coordinates)
                single_route = [0] + list(range(1, n)) + [0]
                total_distance_single_km = sum(dist_m[single_route[i]][single_route[i + 1]] / 1000 for i in range(len(single_route) - 1))
                fuel_liters_single = (total_distance_single_km / 100) * config["default_settings"]["fuel_consumption"]
                emission_factor = 2.68 if config["default_settings"]["fuel_type"] == "diesel" else 2.31
                co2_single_kg = fuel_liters_single * emission_factor
                savings = (co2_single_kg - total_co2_optimized) / co2_single_kg * 100 if co2_single_kg > 0 else 0
                st.metric(
                    label="🌍 Ahorro de CO2",
                    value=f"{savings:.2f}%",
                    help="Porcentaje de reducción de emisiones respecto a una ruta no optimizada."
                )

        # Análisis por Vehículo
        st.subheader("🚗 Análisis por Vehículo", anchor=False)
        with st.container(border=True):
            if not st.session_state["kpi_df"].empty:
                st.dataframe(
                    st.session_state["kpi_df"],
                    use_container_width=True,
                    column_config={
                        "Vehículo": st.column_config.TextColumn("Vehículo"),
                        "Kilómetros Recorridos": st.column_config.NumberColumn("Kilómetros (km)", format="%.2f"),
                        "Tiempo Total (min)": st.column_config.NumberColumn("Tiempo (min)", format="%.2f"),
                        "Coste Total de la Ruta (€)": st.column_config.NumberColumn("Coste (€)", format="%.2f"),
                        "Paradas": st.column_config.NumberColumn("Paradas")
                    }
                )
                excel_kpi_buffer = io.BytesIO()
                st.session_state["kpi_df"].to_excel(excel_kpi_buffer, index=False, engine="openpyxl")
                st.download_button(
                    label="📥 Descargar KPIs (Excel)",
                    data=excel_kpi_buffer,
                    file_name=f"kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_kpis_excel",
                    use_container_width=True
                )
            else:
                st.info("No hay datos de KPIs para mostrar.")
                logging.warning("kpi_df vacío")

        # Itinerario de Vehículos
        st.subheader("🗺️ Itinerario de Vehículos", anchor=False)
        with st.container(border=True):
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
                            eta_str = f"{eta_val // 60:02d}:{eta_val % 60:02d}" if eta_val is not None else "N/A"
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
                        key="download_plan_excel",
                        use_container_width=True
                    )
            else:
                st.info("No hay itinerarios de vehículos para mostrar.")

        # Links para Conductores
        st.subheader("🔗 Links para Conductores", anchor=False)
        with st.container(border=True):
            if "links" in st.session_state and st.session_state["links"] is not None and not st.session_state["links"].empty:
                st.dataframe(
                    st.session_state["links"],
                    use_container_width=True,
                    column_config={
                        "Link": st.column_config.LinkColumn("Link", display_text="Abrir Ruta")
                    }
                )
            else:
                st.info("No hay links para conductores disponibles.")

        # Mapa de Rutas
        st.subheader("🗺️ Mapa de Rutas", anchor=False)
        with st.container(border=True):
            if st.session_state["map"]:
                st_folium(st.session_state["map"], use_container_width=True, height=600, key="route_map")
                html_buffer = io.BytesIO()
                st.session_state["map"].save(html_buffer, close_file=False)
                html_buffer.seek(0)
                st.download_button(
                    label="📥 Descargar Mapa (HTML)",
                    data=html_buffer,
                    file_name=f"mapa_rutas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    key="download_map",
                    use_container_width=True
                )
            else:
                st.info("No hay mapa disponible para mostrar.")

        # Eficiencia de Rutas
        st.subheader("📉 Eficiencia de Rutas", anchor=False)
        with st.container(border=True):
            total_distance_optimized = sum(st.session_state["kpi_df"]["Kilómetros Recorridos"]) if not st.session_state["kpi_df"].empty else 0
            efficiency_score = (total_distance_single_km - total_distance_optimized) / total_distance_single_km * 100 if total_distance_single_km > 0 else 0
            st.metric(
                label="Puntuación de Eficiencia",
                value=f"{efficiency_score:.2f}%",
                help="Porcentaje de reducción en distancia respecto a una ruta no optimizada."
            )

if __name__ == "__main__":
    standard_planner()