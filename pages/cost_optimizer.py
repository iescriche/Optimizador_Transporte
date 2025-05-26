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
import altair as alt
import math
import os
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

# Constants
MAX_H_REG = 8  # Regular hours limit

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
            st.session_state[key] = None

def get_cached_matrix(coords, api_key):
    """Obtiene matrices de distancia/tiempo desde caché o API."""
    print("Iniciando get_cached_matrix...")
    print(f"Número de coordenadas: {len(coords)}")
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

def main():
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
        logging.error(f"Error en config.yaml: {e}")
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
                    logging.error(f"Error en carga automática: {e}")
                    return
            elif mode == "Subir archivo CSV":
                up_file = st.file_uploader("Subir archivo CSV", type=["csv"], key="cost_csv_upload")
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
                up_cli = st.file_uploader("Maestro clientes (xlsx/csv)", type=["xlsx", "xls", "csv"], key="cost_clients_upload")
                up_rta = st.file_uploader("Rutas del día (xlsx/csv)", type=["xlsx", "xls", "csv"], key="cost_routes_upload")
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
                    logging.error(f"Error procesando datos: {e}")
                    return
            
            if df_today.empty:
                st.warning("No se cargaron datos válidos.")
                logging.warning("df_today vacío")
                return
            
            if not validate_coordinates(df_today):
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
            print(f"Depósito seleccionado: {depot_address}, índice original: {depot_idx}")
            
            # Reordenar df_today para que el depósito esté en el índice 0
            df_today = pd.concat([df_today.loc[[depot_idx]], df_today.drop(depot_idx)]).reset_index(drop=True)
            depot_idx = 0  # Después de reordenar, el depósito siempre está en el índice 0
            print(f"Depósito reordenado al índice: {depot_idx}, dirección: {df_today.at[depot_idx, 'DIRECCION']}")

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
                vehs = st.slider("Furgonetas máximas disponibles", min_value=1, max_value=10, value=1, key="cost_vehicles")
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
            start_time_minutes = start_time.hour * 60 + start_time.minute
            service_time = st.number_input("Tiempo de servicio por parada (min)", min_value=0, value=config["default_settings"]["service_time"], key="cost_service_time")
            balance_threshold = st.slider("Umbral de balanceo (%)", 50, 100, 90, key="cost_balance_threshold") / 100
            max_stops_per_vehicle = st.number_input(
                "📍 Máximo de paradas por vehículo (0 para sin límite)",
                min_value=0,
                value=0,
                help="Define el número máximo de paradas por vehículo. Usa 0 para no limitar.",
                key="cost_max_stops"
            )
            recomendar = st.checkbox("Recomendar número óptimo de vehículos", value=True, key="cost_recomendar")

        with st.expander("💶 Costes"):
            price_per_hour = st.number_input(
                "Sueldo por hora del repartidor (€/hora)", min_value=0.0, value=10.0, step=0.1, key="cost_price_per_hour"
            )
            extra_max = st.number_input(
                "Máx. horas extra por conductor", min_value=0.0, max_value=12.0, step=0.5, value=1.0, key="cost_extra_max_h"
            )
            extra_price = st.number_input(
                "Precio de la hora extra (€/h)", min_value=0.0, step=0.1, value=price_per_hour * 1.5, key="cost_extra_price_h"
            )
            max_minutes = int((MAX_H_REG + extra_max) * 60)

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
                print("Error: Clave ORS inválida o no proporcionada.")
                return

            if df_today.empty or not all(col in df_today.columns for col in ["DIRECCION", "LATITUD", "LONGITUD"]):
                st.error("Datos inválidos. Asegúrate de que el archivo contiene las columnas 'DIRECCION', 'LATITUD', 'LONGITUD'.")
                logging.error("Datos inválidos o faltan columnas")
                print("Error: Datos inválidos o faltan columnas.")
                return

            try:
                with st.spinner("Calculando rutas óptimas..."):
                    logging.info("Iniciando cálculo de rutas")
                    print("Iniciando cálculo de rutas...")
                    coords = list(zip(df_today["LATITUD"], df_today["LONGITUD"]))
                    print(f"Coordenadas: {coords}")
                    dist_m_global, time_m_global = get_cached_matrix(coords, api_key)
                    print(f"Después de get_cached_matrix: dist_m tipo={type(dist_m_global)}, time_m tipo={type(time_m_global)}")
                    if dist_m_global is None or time_m_global is None:
                        st.error("No se pudieron obtener las matrices de distancia y tiempo.")
                        logging.error("Matrices inválidas: dist_m=%s, time_m=%s", dist_m_global, time_m_global)
                        print("Error: No se pudieron obtener las matrices.")
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

                    # Extraer ventanas de tiempo
                    time_windows = None
                    if "TIME_WINDOW_START" in df_today.columns and "TIME_WINDOW_END" in df_today.columns:
                        time_windows = list(zip(df_today["TIME_WINDOW_START"], df_today["TIME_WINDOW_END"]))
                        print(f"Ventanas de tiempo: {time_windows}")
                    else:
                        time_windows = [(0, 1440)] * len(df_today)  # Default: sin restricciones
                        print("No se proporcionaron ventanas de tiempo, usando [0, 1440] para todos los nodos.")

                    n_stops = len(coords) - 1
                    fixed_hours = 2
                    fixed_cents = int(round(fixed_hours * price_per_hour * 100))

                    resultados = []
                    if respect_predefined and predefined_routes:
                        routes, eta, used = solve_vrp_simple(
                            dist_m=dist_m_global,
                            time_m=time_m_global,
                            vehicles=vehs,
                            depot=depot_idx,
                            balance=balance,
                            start_min=start_time_minutes,
                            service_time=service_time,
                            time_windows=time_windows,
                            max_stops_per_vehicle=math.ceil(n_stops / vehs) + 2 if max_stops_per_vehicle == 0 else max_stops_per_vehicle,
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
                            st.error("No se encontraron rutas válidas para las rutas predefinidas.")
                            return
                        if reassign_stops:
                            routes, eta, used = reassign_nearby_stops(
                                routes=routes,
                                dist_m=dist_m_global,
                                time_m=time_m_global,
                                depot=depot_idx,
                                balance=balance,
                                start_min=start_time_minutes,
                                service_time=service_time,
                                time_windows=time_windows,
                                max_stops_per_vehicle=math.ceil(n_stops / vehs) + 2 if max_stops_per_vehicle == 0 else max_stops_per_vehicle,
                                max_distance_m=reassign_distance,
                                balance_threshold=balance_threshold,
                                fuel_price=fuel_price,
                                fuel_consumption=fuel_consumption,
                                price_per_hour=price_per_hour,
                                max_minutes=max_minutes
                            )
                        eta = recompute_etas(routes, time_m_global, start_time_minutes, service_time, len(df_today), time_windows)
                        kpi_df, km_per_order, euro_per_order = calculate_kpis(
                            plan=routes,
                            dist_m=dist_m_global,
                            time_m=time_m_global,
                            df_today=df_today,
                            start_time_minutes=start_time_minutes,
                            service_time=service_time,
                            price_per_hour=price_per_hour if price_per_hour > 0 else None,
                            fuel_price=fuel_price if fuel_price > 0 else None,
                            fuel_consumption=fuel_consumption if fuel_consumption > 0 else None
                        )
                        coste_variable = kpi_df["Coste Total de la Ruta (€)"].sum() if kpi_df["Coste Total de la Ruta (€)"].notna().any() else 0
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
                                depot=depot_idx,
                                balance=balance,
                                start_min=start_time_minutes,
                                service_time=service_time,
                                time_windows=time_windows,
                                max_stops_per_vehicle=math.ceil(n_stops / v) + 2 if max_stops_per_vehicle == 0 else max_stops_per_vehicle,
                                balance_threshold=balance_threshold,
                                fuel_price=fuel_price,
                                fuel_consumption=fuel_consumption,
                                price_per_hour=price_per_hour,
                                max_minutes=max_minutes,
                                cost_mode=True
                            )
                            if not plan:
                                continue
                            eta = recompute_etas(plan, time_m_global, start_time_minutes, service_time, len(df_today), time_windows)
                            kpi_df, km_per_order, euro_per_order = calculate_kpis(
                                plan=plan,
                                dist_m=dist_m_global,
                                time_m=time_m_global,
                                df_today=df_today,
                                start_time_minutes=start_time_minutes,
                                service_time=service_time,
                                price_per_hour=price_per_hour if price_per_hour > 0 else None,
                                fuel_price=fuel_price if fuel_price > 0 else None,
                                fuel_consumption=fuel_consumption if fuel_consumption > 0 else None
                            )
                            coste_variable = kpi_df["Coste Total de la Ruta (€)"].sum() if kpi_df["Coste Total de la Ruta (€)"].notna().any() else 0
                            coste_fijo = v * fixed_cents / 100
                            coste_total = coste_fijo + coste_variable
                            resultados.append((v, coste_total, plan, kpi_df, eta, km_per_order, euro_per_order))

                    if not resultados:
                        st.error("No se encontraron rutas válidas para ningún número de vehículos.")
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
                    fmap = folium.Map(location=df_today[["LATITUD", "LONGITUD"]].mean().tolist(), zoom_start=10)
                    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] * 5
                    polyline_failures = 0
                    links = []
                    for v, rt in enumerate(opt_plan):
                        color = palette[v % len(palette)]
                        for i in range(len(rt) - 1):
                            orig = (df_today.at[rt[i], "LATITUD"], df_today.at[rt[i], "LONGITUD"])
                            dest = (df_today.at[rt[i + 1], "LATITUD"], df_today.at[rt[i + 1], "LONGITUD"])
                            pts = get_polyline_ors(orig, dest, api_key)
                            if pts:
                                folium.PolyLine(pts, color=color, weight=4).add_to(fmap)
                            else:
                                polyline_failures += 1
                                folium.PolyLine([orig, dest], color=color, weight=4, dash_array="5, 5").add_to(fmap)
                                logging.warning(f"Fallo en polilínea para segmento {orig} -> {dest}")
                        for seq, node in enumerate(rt):
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
                        wps = [f"{df_today.at[n, 'LATITUD']},{df_today.at[n, 'LONGITUD']}" for n in rt[1:-1]]
                        url = f"https://www.google.com/maps/dir/{coords[rt[0]][0]},{coords[rt[0]][1]}/{'/'.join(wps)}/{coords[rt[0]][0]},{coords[rt[0]][1]}"
                        links.append({"Vehículo": v, "Link": url})

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
                logging.error(f"Error calculando rutas: {e}")
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
                        "Paradas": st.column_config.NumberColumn("Paradas")
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
                with col_chart3:
                    if st.session_state["kpi_df"]["Tiempo Total (min)"].notna().any():
                        time_chart = plot_kpi_bars(st.session_state["kpi_df"], "Tiempo Total (min)", "Tiempo por Ruta", "Minutos")
                        st.altair_chart(time_chart, use_container_width=True)

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