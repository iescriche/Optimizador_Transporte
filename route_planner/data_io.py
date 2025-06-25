import streamlit as st
import pandas as pd
from typing import Union, Dict
import json
import os
from .utils import time_to_minutes

def load_data(
    clients: Union[str, pd.DataFrame, None],
    routes: Union[str, pd.DataFrame, None],
    config: Dict,
    column_mapping: Dict[str, str] = None
) -> pd.DataFrame:
    """Carga y combina datos de clientes y rutas, aplicando mapeo de columnas."""
    try:
        # Load default mapping if not provided
        mapping_file = "column_mapping.json"
        if column_mapping is None and os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                column_mapping = json.load(f)
        if not column_mapping:
            column_mapping = {}

        # Default column names from config
        default_cols = config.get("default_columns", {})
        col_address = default_cols.get("address", "DIRECCION")
        col_lat = default_cols.get("latitude", "LATITUD")
        col_lon = default_cols.get("longitude", "LONGITUD")
        col_route = column_mapping.get("route", "RUTA")
        col_time_start = column_mapping.get("time_window_start", "TIME_WINDOW_START")
        col_time_end = column_mapping.get("time_window_end", "TIME_WINDOW_END")

        # Load clients
        if isinstance(clients, str):
            clients_df = pd.read_excel(clients) if clients.endswith((".xlsx", ".xls")) else pd.read_csv(clients)
        elif isinstance(clients, pd.DataFrame):
            clients_df = clients.copy()
        else:
            st.error("No se proporcionó archivo de clientes válido.")
            return pd.DataFrame()

        # Load routes
        if isinstance(routes, str):
            routes_df = pd.read_excel(routes) if routes.endswith((".xlsx", ".xls")) else pd.read_csv(routes)
        elif isinstance(routes, pd.DataFrame):
            routes_df = routes.copy()
        else:
            st.error("No se proporcionó archivo de rutas válido.")
            return pd.DataFrame()

        # Apply column mapping
        clients_df = clients_df.rename(columns={
            column_mapping.get("address", col_address): "DIRECCION",
            column_mapping.get("latitude", col_lat): "LATITUD",
            column_mapping.get("longitude", col_lon): "LONGITUD"
        })
        routes_df = routes_df.rename(columns={
            column_mapping.get("address", col_address): "DIRECCION",
            column_mapping.get("route", col_route): "RUTA",
            column_mapping.get("time_window_start", col_time_start): "TIME_WINDOW_START",
            column_mapping.get("time_window_end", col_time_end): "TIME_WINDOW_END"
        })

        # Validate required columns
        required_cols = ["DIRECCION", "LATITUD", "LONGITUD"]
        missing_cols = [col for col in required_cols if col not in clients_df.columns]
        if missing_cols:
            st.error(f"Faltan columnas en clientes: {', '.join(missing_cols)}")
            return pd.DataFrame()

        if "DIRECCION" not in routes_df.columns:
            st.error("Falta columna DIRECCION en rutas")
            return pd.DataFrame()

        # Validate RUTA column if present
        if "RUTA" not in routes_df.columns:
            st.warning("Columna 'RUTA' no encontrada en el archivo de rutas. Creando columna vacía.")
            routes_df["RUTA"] = ""

        # Convert LATITUD and LONGITUD to float
        try:
            clients_df["LATITUD"] = pd.to_numeric(clients_df["LATITUD"], errors="coerce")
            clients_df["LONGITUD"] = pd.to_numeric(clients_df["LONGITUD"], errors="coerce")
        except Exception as e:
            st.error(f"Error al convertir coordenadas a números: {str(e)}")
            return pd.DataFrame()

        # Convert time windows to minutes if present in routes_df
        if "TIME_WINDOW_START" in routes_df.columns and "TIME_WINDOW_END" in routes_df.columns:
            try:
                routes_df["TIME_WINDOW_START"] = routes_df["TIME_WINDOW_START"].apply(lambda x: time_to_minutes(x) if pd.notna(x) else 0)
                routes_df["TIME_WINDOW_END"] = routes_df["TIME_WINDOW_END"].apply(lambda x: time_to_minutes(x) if pd.notna(x) else 1440)
            except Exception as e:
                st.error(f"Error al convertir ventanas de tiempo: {str(e)}")
                return pd.DataFrame()
        else:
            routes_df["TIME_WINDOW_START"] = 0  # Default: sin restricción
            routes_df["TIME_WINDOW_END"] = 1440  # Default: hasta medianoche

        # Merge data
        df_today = clients_df[clients_df["DIRECCION"].isin(routes_df["DIRECCION"])][["DIRECCION", "LATITUD", "LONGITUD"]].merge(
            routes_df[["DIRECCION", "RUTA", "TIME_WINDOW_START", "TIME_WINDOW_END"]], on="DIRECCION", how="left"
        )

        # Validate merged data
        if df_today.empty:
            st.error("No se encontraron direcciones comunes entre clientes y rutas.")
            return pd.DataFrame()

        # Clean coordinates
        df_today = df_today.dropna(subset=["LATITUD", "LONGITUD"])
        df_today = df_today[df_today["LATITUD"].between(-90, 90) & df_today["LONGITUD"].between(-180, 180)]

        # Ensure time windows are filled
        df_today["TIME_WINDOW_START"] = df_today["TIME_WINDOW_START"].fillna(0)
        df_today["TIME_WINDOW_END"] = df_today["TIME_WINDOW_END"].fillna(1440)

        if df_today.empty:
            st.error("No hay datos válidos tras limpiar coordenadas. Verifica que LATITUD y LONGITUD contengan valores numéricos válidos.")

        return df_today.reset_index(drop=True)

    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame()
