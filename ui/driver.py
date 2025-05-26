import streamlit as st
import pandas as pd
import folium
import json
import os
from datetime import datetime
from streamlit_folium import st_folium
from route_planner import solve_vrp_simple, get_polyline_ors, recompute_etas, PlanningResult, ors_matrix_chunk, validate_ors_key

def generate_google_maps_link(lat: float, lng: float) -> str:
    """Generate a Google Maps URL for a single destination."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"

def generate_google_maps_route(route: list, coords: list) -> str:
    """Generate a Google Maps URL for a multi-stop route."""
    waypoints = "/".join(f"{coords[node][0]},{coords[node][1]}" for node in route[1:-1])
    start = f"{coords[route[0]][0]},{coords[route[0]][1]}"
    end = f"{coords[route[-1]][0]},{coords[route[-1]][1]}"
    return f"https://www.google.com/maps/dir/{start}/{waypoints}/{end}"

def main():
    st.set_page_config(page_title="Driver View", page_icon=":truck:", layout="wide")
    st.title("🚚 Vista del Conductor")
    st.markdown("Marca las paradas visitadas y reoptimiza rutas si es necesario.")
    
    uploaded_plan = st.file_uploader("Cargar plan de rutas (JSON)", type=["json"], key="plan_upload")
    if not uploaded_plan:
        st.info("Sube un archivo JSON de un plan generado.")
        return
    
    try:
        plan_data = json.load(uploaded_plan)
        plan = PlanningResult.from_dict(plan_data)
        for v in plan.routes:
            v.route_link = generate_google_maps_route(v.sequence, plan.coordinates)
    except Exception as e:
        st.error(f"Error cargando plan JSON: {e}")
        return
    
    if "plan" not in st.session_state:
        st.session_state["plan"] = plan
    
    # API key input if not provided
    api_key = plan.settings.get("api_key", os.getenv("ORS_API_KEY", ""))
    if not api_key:
        st.warning("Clave ORS no encontrada en el plan o entorno. Por favor, ingresa la clave.")
        api_key = st.text_input("Clave ORS API", type="password", key="driver_api_key")
        if not api_key or not validate_ors_key(api_key):
            st.error("Clave ORS inválida o no proporcionada. Contacta a la oficina.")
            return
    
    st.header("📍 Paradas Asignadas")
    vehicle_id = st.selectbox("Selecciona tu vehículo", [v.vehicle_id for v in plan.routes], key="vehicle_select")
    vehicle = next(v for v in plan.routes if v.vehicle_id == vehicle_id)
    
    df_stops = pd.DataFrame([
        {
            "Índice": node,
            "Dirección": plan.addresses[node],
            "Visitada": vehicle.visited[i],
            "ETA": f"{plan.eta[node] // 60:02d}:{plan.eta[node] % 60:02d}" if plan.eta[node] else "N/A",
            "Enlace Google Maps": generate_google_maps_link(*plan.coordinates[node])
        }
        for i, node in enumerate(vehicle.sequence[1:-1], 1)
    ])
    
    st.subheader("Marcar Paradas Visitadas")
    for idx, row in df_stops.iterrows():
        is_visited = st.checkbox(
            f"Parada {idx}: {row['Dirección']} (ETA: {row['ETA']})",
            value=row["Visitada"],
            key=f"stop_{row['Índice']}_{vehicle_id}"
        )
        vehicle.visited[vehicle.sequence.index(row['Índice'])] = is_visited
    
    st.subheader("Lista de Paradas")
    st.dataframe(df_stops.style.format({"Enlace Google Maps": lambda x: f'<a href="{x}" target="_blank">Abrir</a>'}), use_container_width=True)
    
    st.subheader("Ruta Completa")
    route_link = generate_google_maps_route(vehicle.sequence, plan.coordinates)
    st.button("Abrir Ruta en Google Maps", on_click=lambda: st.markdown(f"[Ruta Vehículo {vehicle_id}]({route_link})"), key="open_route")
    
    fmap = folium.Map(location=plan.coordinates[0], zoom_start=10)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] * 5
    color = palette[vehicle_id % len(palette)]
    polyline_failures = 0
    
    for i in range(len(vehicle.sequence) - 1):
        orig = plan.coordinates[vehicle.sequence[i]]
        dest = plan.coordinates[vehicle.sequence[i + 1]]
        pts = get_polyline_ors(orig, dest, api_key)
        if pts:
            folium.PolyLine(pts, color=color, weight=4).add_to(fmap)
        else:
            polyline_failures += 1
            folium.PolyLine([orig, dest], color=color, weight=4, dash_array="5, 5").add_to(fmap)
        
        eta_str = f"{plan.eta[vehicle.sequence[i]] // 60:02d}:{plan.eta[vehicle.sequence[i]] % 60:02d}" if plan.eta[vehicle.sequence[i]] else "N/A"
        popup_html = f"""
        V{vehicle_id}·{i} {plan.addresses[vehicle.sequence[i]]}<br>
        ETA: {eta_str}<br>
        <a href="{generate_google_maps_link(*plan.coordinates[vehicle.sequence[i]])}" target="_blank">Abrir en Google Maps</a>
        """
        folium.CircleMarker(
            location=plan.coordinates[vehicle.sequence[i]],
            radius=6 if i == 0 else 4,
            color=color,
            fill=True,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(fmap)
    
    if polyline_failures > 0:
        st.warning(f"Se usaron líneas rectas para {polyline_failures} segmentos debido a errores en la API de ORS.")
    
    st.subheader("🗺️ Mapa de Ruta")
    st_folium(fmap, width=1200, height=800, key="driver_map", use_container_width=True)
    
    if st.button("🚚 Estoy lleno → Volver a base y recalcular rutas pendientes", key="reoptimize_route"):
        pending_stops = [i for i, v in enumerate(vehicle.visited[1:-1], 1) if not v]
        if not pending_stops:
            st.info("Todas las paradas están visitadas.")
            return
        
        if st.checkbox("Confirmar reoptimización de ruta", key="confirm_reoptimize"):
            with st.spinner("Reoptimizando ruta..."):
                unloading_time = st.session_state.get("unloading_time", 15)
                
                last_visited_idx = max([i for i, v in enumerate(vehicle.visited[1:-1], 1) if v], default=0)
                current_node = vehicle.sequence[last_visited_idx] if last_visited_idx > 0 else vehicle.sequence[0]
                
                depot_coords = plan.coordinates[0]
                current_coords = plan.coordinates[current_node]
                dist_m_phase1, time_m_phase1 = ors_matrix_chunk([current_coords, depot_coords], api_key)
                
                pending_indices = [vehicle.sequence[i] for i in pending_stops]
                all_indices = [0] + pending_indices
                pending_coords = [plan.coordinates[i] for i in all_indices]
                dist_m_phase2, time_m_phase2 = ors_matrix_chunk(pending_coords, api_key)
                
                routes, eta, used = solve_vrp_simple(
                    dist_m=dist_m_phase2,
                    time_m=time_m_phase2,
                    vehicles=1,
                    depot=0,
                    balance=False,
                    start_min=plan.settings["start_time"],
                    service_time=plan.settings["service_time"],
                    max_stops_per_vehicle=None
                )
                eta = recompute_etas(routes, time_m_phase2, plan.settings["start_time"] + time_m_phase1[0][1] + unloading_time, plan.settings["service_time"], len(pending_coords))
                
                new_sequence = [current_node, 0] + [pending_indices[i - 1] for i in routes[0][1:-1]] + [0]
                vehicle.sequence = new_sequence
                vehicle.visited = [False] * len(new_sequence)
                
                plan.eta = [None] * len(plan.coordinates)
                plan.eta[current_node] = plan.settings["start_time"]
                plan.eta[0] = plan.settings["start_time"] + time_m_phase1[0][1] + unloading_time
                for i, e in enumerate(eta[1:], 1):
                    if i - 1 < len(pending_indices):
                        plan.eta[pending_indices[i - 1]] = e
                
                vehicle.route_link = generate_google_maps_route(new_sequence, plan.coordinates)
                plan_file = f"plans/{datetime.now().strftime('%Y-%m-%d')}_plan_{uuid.uuid4().hex[:8]}.json"
                os.makedirs("plans", exist_ok=True)
                with open(plan_file, "w") as f:
                    json.dump(plan.asdict(), f, indent=2)
                
                st.success("Ruta reoptimizada con retorno a base y guardada.")
                st.session_state["plan"] = plan
                st.experimental_rerun()

if __name__ == "__main__":
    main()