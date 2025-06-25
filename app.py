import streamlit as st
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="🚚 Route Optimizer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown(
    """
    <style>
    /* Fondo y tipografía general */
    .main-container {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }

    /* Título principal */
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Texto descriptivo */
    .description {
        text-align: center;
        font-size: 1.1rem;
        color: #7f8c8d;
        line-height: 1.6;
        padding: 0 10%;
        margin-bottom: 30px;
    }

    /* Tarjetas de módulos */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin: 10px;
        transition: transform 0.2s;
        cursor: pointer;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .card-title {
        color: #2980b9;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .card-text {
        color: #34495e;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Botones */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }

    /* Pie de página */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #ecf0f1;
    }

    /* Responsividad */
    @media (max-width: 768px) {
        .title { font-size: 2rem; }
        .description { font-size: 1rem; padding: 0 5%; }
        .card { margin: 5px; padding: 15px; }
        .card-title { font-size: 1.2rem; }
        .card-text { font-size: 0.9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Contenedor principal
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Título principal
    st.markdown("<div class='title'>🚚 Route Optimizer</div>", unsafe_allow_html=True)

    # Descripción introductoria
    st.markdown(
        """
        <div class='description'>
            Bienvenido a <b>Route Optimizer</b>, tu solución integral para planificar y optimizar rutas de entrega.<br>
            Potenciada por <b>OpenRouteService</b>, esta aplicación te ayuda a minimizar costos y maximizar la eficiencia operativa.<br>
            Explora los módulos a continuación o usa la barra lateral para navegar.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Separador
    st.markdown("---")

    # Tarjetas en columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>🧮 Cost Optimizer</div>
                <div class='card-text'>
                    - Minimiza costos logísticos.<br>
                    - Incluye salarios y combustible.<br>
                    - Análisis detallados.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Ir a Cost Optimizer", key="go_cost_optimizer"):
            try:
                st.switch_page("pages/cost_optimizer.py")
            except Exception as e:
                st.error(f"Error: {e}. Verifica que 'pages/cost_optimizer.py' exista.")

    with col2:
        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>📍 Driver View</div>
                <div class='card-text'>
                    - Marca paradas visitadas.<br>
                    - Reoptimiza rutas en tiempo real.<br>
                    - Mapas interactivos.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Ir a Driver View", key="go_driver_view"):
            try:
                st.switch_page("pages/driver_view.py")
            except Exception as e:
                st.error(f"Error: {e}. Verifica que 'pages/driver_view.py' exista.")

    with col3:
        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>🛤️ Route Planner</div>
                <div class='card-text'>
                    - Crea rutas balanceadas.<br>
                    - Optimiza tiempos y distancias.<br>
                    - Ideal para planificación diaria.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Ir a Route Planner", key="go_route_planner"):
            try:
                st.switch_page("pages/route_planner.py")
            except Exception as e:
                st.error(f"Error: {e}. Verifica que 'pages/route_planner.py' exista.")

    # Pie de página
    st.markdown(
        f"""
        <div class='footer'>
            Route Optimizer v1.0 | Desarrollado por xAI | © {datetime.now().year} Todos los derechos reservados.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)