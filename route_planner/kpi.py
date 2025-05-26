import pandas as pd
import streamlit as st
import logging
from typing import List, Tuple, Optional

# Configurar logging
logging.basicConfig(filename="route_planner.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    optimize_for: str = "Coste"
) -> Tuple[pd.DataFrame, float, float]:
    """Calcula KPIs: km/ruta, km/pedido, euro/ruta, euro/pedido."""
    logging.info("Iniciando cálculo de KPIs")
    
    # Validar matrices
    n = len(df_today)
    if not (isinstance(dist_m, list) and isinstance(time_m, list) and
            len(dist_m) == n and len(time_m) == n and
            all(isinstance(row, list) and len(row) == n for row in dist_m) and
            all(isinstance(row, list) and len(row) == n for row in time_m)):
        st.error("Matriz de distancias o tiempos inválida.")
        logging.error("Matrices inválidas: dist_m=%s, time_m=%s", type(dist_m), type(time_m))
        raise ValueError("Matriz de distancias o tiempos no es una lista bidimensional válida.")

    if not plan:
        st.warning("El plan de rutas está vacío.")
        return pd.DataFrame(), 0, 0

    # Resto del código sin cambios...

    kpi_data = []
    total_distance = 0
    total_time = 0
    total_cost = 0
    total_stops = 0

    for v, route in enumerate(plan):
        if len(route) <= 2:
            continue
        try:
            route_distance = sum(dist_m[route[i]][route[i + 1]] / 1000 for i in range(len(route) - 1))
            stops = len(route) - 2
            if stops == 0:
                continue

            # Calcular tiempo total como última ETA (excluyendo depósito final) menos start_time_minutes
            last_stop_idx = route[-2] if len(route) > 2 else route[-1]  # Última parada antes del depósito
            route_time = (eta[last_stop_idx] - start_time_minutes) if eta[last_stop_idx] is not None else 0
            route_time = max(route_time, 0)  # Evitar tiempos negativos

            route_cost = 0
            if optimize_for == "Coste" and fuel_price is not None and fuel_consumption is not None and fuel_price > 0 and fuel_consumption > 0:
                fuel_cost = (route_distance / 100) * fuel_consumption * fuel_price
                route_cost += fuel_cost
            if optimize_for == "Coste" and price_per_hour is not None and price_per_hour > 0:
                driver_cost = route_time * price_per_hour / 60  # Convertir minutos a horas
                route_cost += driver_cost

            kpi_data.append({
                "Vehículo": f"Vehículo {v + 1}",
                "Kilómetros Recorridos": round(route_distance, 2),
                "Tiempo Total (min)": round(route_time, 2),
                "Coste Total de la Ruta (€)": round(route_cost, 2) if route_cost > 0 and optimize_for == "Coste" else 0,
                "Paradas": stops
            })

            total_distance += route_distance
            total_time += route_time
            total_cost += route_cost if route_cost > 0 and optimize_for == "Coste" else 0
            total_stops += stops
            logging.info(f"KPIs calculados para Vehículo {v + 1}: Distancia={route_distance:.2f}km, Tiempo={route_time:.2f}min, Coste={route_cost:.2f}€")
        except IndexError as e:
            st.error(f"Error al calcular KPIs para Vehículo {v + 1}: {e}")
            logging.error(f"IndexError en Vehículo {v + 1}: {e}")
            continue

    kpi_data.append({
        "Vehículo": "Total",
        "Kilómetros Recorridos": round(total_distance, 2),
        "Tiempo Total (min)": round(total_time, 2),
        "Coste Total de la Ruta (€)": round(total_cost, 2) if total_cost > 0 and optimize_for == "Coste" else 0,
        "Paradas": total_stops
    })

    kpi_df = pd.DataFrame(kpi_data)
    km_per_order = total_distance / total_stops if total_stops > 0 else 0
    time_per_order = total_time / total_stops if total_stops > 0 else 0
    euro_per_order = total_cost / total_stops if total_stops > 0 and total_cost > 0 and optimize_for == "Coste" else 0
    logging.info(f"KPIs globales: km_per_order={km_per_order:.2f}, time_per_order={time_per_order:.2f}min, euro_per_order={euro_per_order:.2f}")

    return kpi_df, km_per_order, euro_per_order