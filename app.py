"""Interfaz web (Streamlit) para calcular áreas de superficies de revolución.

Ejecutar con:  streamlit run app.py
"""

from __future__ import annotations

import math

import pandas as pd
import sympy as sp
import streamlit as st

from revolucion import graficas, integracion
from revolucion.calculo import ResultadoSuperficie, calcular_superficie
from revolucion.ejemplos import EJEMPLOS
from revolucion.excepciones import SuperficieError
from revolucion.validacion import EJE_X, EJES

PERSONALIZADO = "Personalizado"


def inicializar_estado() -> None:
    """Valores por defecto de los controles (esfera de radio 2)."""
    esfera = next(iter(EJEMPLOS.values()))
    valores = {
        "ejemplo": esfera.nombre,
        "funcion": esfera.funcion,
        "a": esfera.a,
        "b": esfera.b,
        "eje": esfera.eje,
        "metodo": integracion.SIMPSON,
        "n": 100,
        "resultado": None,
        "error": None,
    }
    for clave, valor in valores.items():
        st.session_state.setdefault(clave, valor)


def cargar_ejemplo() -> None:
    """Copia el ejemplo elegido a los controles del sidebar."""
    ejemplo = EJEMPLOS.get(st.session_state.ejemplo)
    if ejemplo is None:
        return
    st.session_state.funcion = ejemplo.funcion
    st.session_state.a = ejemplo.a
    st.session_state.b = ejemplo.b
    st.session_state.eje = ejemplo.eje
    st.session_state.resultado = None
    st.session_state.error = None


def area_exacta_actual() -> tuple[float | None, str | None]:
    """Área exacta si los controles coinciden con un ejemplo precargado."""
    ejemplo = EJEMPLOS.get(st.session_state.ejemplo)
    if ejemplo is None:
        return None, None
    coincide = (
        st.session_state.funcion.replace(" ", "") == ejemplo.funcion.replace(" ", "")
        and st.session_state.a == ejemplo.a
        and st.session_state.b == ejemplo.b
        and st.session_state.eje == ejemplo.eje
    )
    return (ejemplo.area_exacta, ejemplo.formula) if coincide else (None, None)


def sidebar() -> None:
    """Controles de entrada."""
    with st.sidebar:
        st.header("Parámetros")
        st.selectbox(
            "Ejemplo precargado",
            [PERSONALIZADO, *EJEMPLOS.keys()],
            key="ejemplo",
            on_change=cargar_ejemplo,
        )
        st.text_input(
            "f(x)",
            key="funcion",
            help="Ejemplos: sqrt(4 - x^2), 2x + 1, x^2, exp(-x), sin(x) + 2. "
            "Se admite multiplicación implícita (2x) y ^ como potencia.",
        )
        col_a, col_b = st.columns(2)
        col_a.number_input("a", key="a", format="%.3f", step=0.5)
        col_b.number_input("b", key="b", format="%.3f", step=0.5)
        st.radio("Eje de rotación", EJES, key="eje", horizontal=True)
        st.selectbox("Método", integracion.METODOS, key="metodo")
        st.number_input(
            "n (subintervalos)",
            min_value=1,
            max_value=1_000_000,
            step=2,
            key="n",
            help="Simpson 1/3 requiere n par.",
        )


def calcular() -> None:
    """Ejecuta el cálculo y guarda el resultado (o el error) en la sesión."""
    exacta, _ = area_exacta_actual()
    try:
        with st.spinner("Analizando la función e integrando..."):
            st.session_state.resultado = calcular_superficie(
                st.session_state.funcion,
                st.session_state.a,
                st.session_state.b,
                st.session_state.eje,
                st.session_state.metodo,
                int(st.session_state.n),
                area_exacta=exacta,
            )
        st.session_state.error = None
    except SuperficieError as exc:
        st.session_state.resultado = None
        st.session_state.error = str(exc)


def mostrar_formula(res: ResultadoSuperficie) -> None:
    """Fórmula de la integral con f y f' simbólicas."""
    an = res.analisis
    radio = r"\left|f(x)\right|" if an.eje == EJE_X else r"\left|x\right|"
    st.latex(
        rf"S = 2\pi \int_{{{an.a:g}}}^{{{an.b:g}}} {radio}\,"
        r"\sqrt{1 + f'(x)^2}\,dx"
    )
    c1, c2 = st.columns(2)
    c1.latex(rf"f(x) = {sp.latex(an.expr)}")
    c2.latex(rf"f'(x) = {sp.latex(sp.simplify(an.derivada))}")


def mostrar_resultado(res: ResultadoSuperficie) -> None:
    """Métricas, errores, tabla y gráficas."""
    for aviso in res.analisis.advertencias:
        st.warning(aviso)

    mostrar_formula(res)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Área ({res.metodo}, n = {res.n})", f"{res.area:.10f}")
    c2.metric(
        "Referencia scipy.quad",
        f"{res.area_referencia:.10f}",
        help=f"Error estimado por quad: {res.error_quad:.2e}",
    )
    etiqueta = "exacta" if res.area_exacta is not None else "quad"
    c3.metric(f"Error absoluto (vs {etiqueta})", f"{res.error_absoluto:.3e}")
    c4.metric(f"Error relativo (vs {etiqueta})", f"{res.error_relativo:.3e}")

    detalles = []
    if res.area_exacta is not None:
        _, formula = area_exacta_actual()
        detalles.append(f"Área exacta: **{res.area_exacta:.10f}** ({formula}).")
    if res.error_richardson is not None:
        detalles.append(
            f"Error estimado por Richardson (|Iₙ − Iₙ/₂| / (2ᵖ − 1)): "
            f"**{res.error_richardson:.3e}**."
        )
    detalles.append(f"Área en múltiplos de π: **{res.area / math.pi:.8f} π**.")
    st.info("  \n".join(detalles))

    tab_2d, tab_3d, tab_conv = st.tabs(["Curva 2D", "Sólido 3D", "Convergencia"])
    an = res.analisis
    with tab_2d:
        st.plotly_chart(graficas.grafica_curva(an.f, an.a, an.b, an.eje), use_container_width=True)
    with tab_3d:
        st.plotly_chart(graficas.grafica_solido(an.f, an.a, an.b, an.eje), use_container_width=True)
    with tab_conv:
        st.caption(
            f"Errores medidos contra el valor "
            f"{'exacto' if res.area_exacta is not None else 'de scipy.quad'}. El orden observado "
            "log₂(error(n/2) / error(n)) debería acercarse a 2 (trapecio) y 4 "
            "(Simpson) cuando la función es suave."
        )
        st.dataframe(tabla_convergencia(res), use_container_width=True, hide_index=True)
        st.plotly_chart(graficas.grafica_convergencia(res.convergencia), use_container_width=True)


def tabla_convergencia(res: ResultadoSuperficie) -> pd.DataFrame:
    """Tabla de convergencia formateada para mostrar."""

    def orden(valor: float | None) -> str:
        return "—" if valor is None else f"{valor:.3f}"

    return pd.DataFrame(
        {
            "n": [f.n for f in res.convergencia],
            "Trapecio": [f"{f.trapecio:.10f}" for f in res.convergencia],
            "Error trapecio": [f"{f.error_trapecio:.3e}" for f in res.convergencia],
            "Orden trapecio": [orden(f.orden_trapecio) for f in res.convergencia],
            "Simpson": [f"{f.simpson:.10f}" for f in res.convergencia],
            "Error Simpson": [f"{f.error_simpson:.3e}" for f in res.convergencia],
            "Orden Simpson": [orden(f.orden_simpson) for f in res.convergencia],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Superficie de revolución", page_icon="🌀", layout="wide")
    inicializar_estado()
    sidebar()

    st.title("🌀 Área de superficies de revolución")
    st.write(
        "Escribe una función f(x), elige el intervalo y el eje de giro, y calcula "
        "el área numéricamente con **Simpson 1/3** o **trapecio compuesto** "
        "(implementados a mano), comparando con `scipy.integrate.quad`."
    )

    if st.button("Calcular", type="primary"):
        calcular()

    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.resultado is not None:
        mostrar_resultado(st.session_state.resultado)


if __name__ == "__main__":
    from streamlit import runtime

    if runtime.exists():
        main()
    else:
        # Permite arrancar también con "python app.py". El modo headless evita
        # la pregunta del correo de la primera ejecución de Streamlit, así que
        # el navegador se abre a mano.
        import sys
        import threading
        import webbrowser

        from streamlit.web import cli

        threading.Timer(3, webbrowser.open, args=["http://localhost:8501"]).start()
        sys.argv = ["streamlit", "run", __file__, "--server.headless", "true", *sys.argv[1:]]
        sys.exit(cli.main())
