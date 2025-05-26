import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="🚚 Route Optimizer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Estilo CSS personalizado
    st.markdown(
        """
        <style>
        /* Fondo y tipografía general */
        .main {
            background-color: #f5f7fa;
            padding: 20px;
            border-radius: 10px;
        }
        .stApp {
            background-color: #ffffff;
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

        /* Subtítulos */
        .subtitle {
            color: #34495e;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 10px;
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
            .title {
                font-size: 2rem;
            }
            .description {
                font-size: 1rem;
                padding: 0 5%;
            }
            .card {
                margin: 5px;
                padding: 15px;
            }
            .card-title {
                font-size: 1.2rem;
            }
            .card-text {
                font-size: 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Contenedor principal
    with st.container():
        # Título principal
        st.markdown(
            "<div class='title'>🚚 Route Optimizer</div>",
            unsafe_allow_html=True
        )

        # Descripción introductoria
        st.markdown(
            """
            <div class='description'>
                Bienvenido a <b>Route Optimizer</b>, tu solución integral para planificar y optimizar rutas de entrega.<br>
                Potenciada por <b>OpenRouteService</b>, esta aplicación te ayuda a minimizar costos y maximizar la eficiencia operativa.<br>
                Explora nuestros módulos en la barra lateral para comenzar.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Separador
        st.markdown("---")

        # Tarjetas de módulos en columnas
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown(
                """
                <div class='card'>
                    <div class='card-title'>🧮 Cost Optimizer</div>
                    <div class='card-text'>
                        - Minimiza costos logísticos considerando salarios, horas extra y combustible.<br>
                        - Genera análisis detallados para una planificación estratégica.<br>
                        - Ideal para optimizar presupuestos y recursos.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Ir a Cost Optimizer", key="go_cost_optimizer"):
                st.switch_page("pages/cost_optimizer.py")

        with col2:
            st.markdown(
                """
                <div class='card'>
                    <div class='card-title'>📍 Route Planner</div>
                    <div class='card-text'>
                        - Crea rutas balanceadas y eficientes para múltiples vehículos.<br>
                        - Reasigna paradas cercanas para mejorar la logística diaria.<br>
                        - Perfecto para operaciones de entrega en tiempo real.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Ir a Route Planner", key="go_route_planner"):
                st.switch_page("pages/route_planner.py")

        # Pie de página
        st.markdown(
            """
            <div class='footer'>
                Route Optimizer v1.0 | Desarrollado por xAI | <a href="mailto:support@routeoptimizer.com">Contacto</a> | 
                <a href="https://docs.routeoptimizer.com" target="_blank">Documentación</a><br>
                © 2025 Todos los derechos reservados.
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()