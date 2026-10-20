"""Gráficas con Plotly: curva 2D, sólido 3D y convergencia."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from revolucion.integracion import FilaConvergencia
from revolucion.parser import FuncionNumerica
from revolucion.validacion import EJE_X


def grafica_curva(f: FuncionNumerica, a: float, b: float, eje: str) -> go.Figure:
    """Curva y = f(x) en [a, b] con el eje de rotación resaltado."""
    xs = np.linspace(a, b, 600)
    ys = f(xs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(width=3, color="#1f77b4"))
    )
    if eje == EJE_X:
        fig.add_trace(
            go.Scatter(x=xs, y=-ys, mode="lines", name="−f(x) (reflejo)",
                       line=dict(width=1, dash="dot", color="#1f77b4"))
        )
        fig.add_hline(y=0, line=dict(color="crimson", width=2), annotation_text="eje de giro")
    else:
        fig.add_trace(
            go.Scatter(x=-xs, y=ys, mode="lines", name="reflejo",
                       line=dict(width=1, dash="dot", color="#1f77b4"))
        )
        fig.add_vline(x=0, line=dict(color="crimson", width=2), annotation_text="eje de giro")

    fig.update_layout(
        title="Curva generatriz",
        xaxis_title="x",
        yaxis_title="y",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def grafica_solido(
    f: FuncionNumerica, a: float, b: float, eje: str, resolucion: int = 120
) -> go.Figure:
    """Superficie de revolución 3D interactiva.

    - Eje x: (x, f(x)·cos θ, f(x)·sin θ)
    - Eje y: (x·cos θ, f(x), x·sin θ)
    """
    xs = np.linspace(a, b, resolucion)
    theta = np.linspace(0, 2 * np.pi, resolucion)
    x_m, t_m = np.meshgrid(xs, theta)
    f_m = f(x_m.ravel()).reshape(x_m.shape)

    if eje == EJE_X:
        X, Y, Z = x_m, f_m * np.cos(t_m), f_m * np.sin(t_m)
        titulos = dict(xaxis_title="x (eje de giro)", yaxis_title="y", zaxis_title="z")
    else:
        X, Y, Z = x_m * np.cos(t_m), f_m, x_m * np.sin(t_m)
        titulos = dict(xaxis_title="x", yaxis_title="y (eje de giro)", zaxis_title="z")

    fig = go.Figure(
        go.Surface(
            x=X, y=Y, z=Z, surfacecolor=x_m, colorscale="Viridis",
            showscale=False, opacity=0.95,
        )
    )
    fig.update_layout(
        title="Sólido de revolución",
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(aspectmode="data", **titulos),
    )
    return fig


def grafica_convergencia(filas: list[FilaConvergencia]) -> go.Figure:
    """Error contra n en escala log-log para ambos métodos."""
    n = [fila.n for fila in filas]
    # Los errores nulos no se pueden dibujar en escala logarítmica.
    err_t = [fila.error_trapecio or None for fila in filas]
    err_s = [fila.error_simpson or None for fila in filas]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=err_t, mode="lines+markers", name="Trapecio (≈ h²)"))
    fig.add_trace(go.Scatter(x=n, y=err_s, mode="lines+markers", name="Simpson 1/3 (≈ h⁴)"))
    fig.update_layout(
        title="Convergencia del error",
        xaxis=dict(title="n (subintervalos)", type="log"),
        yaxis=dict(title="Error absoluto", type="log", exponentformat="e"),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
