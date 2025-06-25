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
            continue
        try:
            # Validar índices de la ruta
            if not all(isinstance(i, int) and 0 <= i < n for i in route):
                logging.error(f"Índices inválidos en ruta del Vehículo {v + 1}: {route}")
                st.error(f"Error: Ruta del Vehículo {v + 1} contiene índices inválidos.")
                continue

            route_distance = sum(dist_m[route[i]][route[i + 1]] / 1000 for i in range(len(route) - 1))
            stops = len(route) - 2
            if stops == 0:
                logging.info(f"Sin paradas válidas para Vehículo {v + 1}, omitiendo")
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

            route_cost = 0
            if optimize_for == "Coste" and fuel_price is not None and fuel_consumption is not None and fuel_price > 0 and fuel_consumption > 0:
                fuel_cost = (route_distance / 100) * fuel_consumption * fuel_price
                route_cost += fuel_cost
            if optimize_for == "Coste" and price_per_hour is not None and price_per_hour > 0:
                # Calcular horas regulares y extras
                regular_hours = min(route_time_hours, MAX_H_REG)
                extra_hours = max(0, route_time_hours - MAX_H_REG)
                if extra_max is not None:
                    extra_hours = min(extra_hours, extra_max)  # Limitar horas extras
                driver_cost = regular_hours * price_per_hour
                if extra_price is not None and extra_hours > 0:
                    driver_cost += extra_hours * extra_price
                else:
                    driver_cost += extra_hours * price_per_hour * 1.5  # Tarifa por defecto para horas extras (50% más)
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
        "Paradas": total_stops
    })

    kpi_df = pd.DataFrame(kpi_data)
    km_per_order = total_distance / total_stops if total_stops > 0 else 0
    time_per_order = total_time / total_stops if total_stops > 0 else 0
    euro_per_order = total_cost / total_stops if total_stops > 0 and total_cost > 0 and optimize_for == "Coste" else 0
    logging.info(f"KPIs globales: km_per_order={km_per_order:.2f}, time_per_order={time_per_order:.2f}min, euro_per_order={euro_per_order:.2f}")

    return kpi_df, km_per_order, euro_per_order
