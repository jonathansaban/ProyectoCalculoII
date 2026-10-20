"""Cálculo completo del área de una superficie de revolución.

Une el parser, la validación y los métodos numéricos en una sola función
``calcular_superficie`` que usa la interfaz (y las pruebas).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

from revolucion import integracion
from revolucion.parser import parsear_funcion
from revolucion.validacion import FuncionAnalizada, analizar_funcion

FACTOR = 2 * math.pi


@dataclass
class ResultadoSuperficie:
    """Todo lo que la interfaz necesita mostrar."""

    analisis: FuncionAnalizada
    metodo: str
    n: int
    area: float
    area_referencia: float
    error_quad: float
    error_absoluto: float
    error_relativo: float
    error_richardson: float | None
    convergencia: list[integracion.FilaConvergencia]
    area_exacta: float | None = None

    @property
    def expr(self) -> sp.Expr:
        return self.analisis.expr


def calcular_superficie(
    texto_funcion: str,
    a: float,
    b: float,
    eje: str,
    metodo: str,
    n: int,
    area_exacta: float | None = None,
    n_max_tabla: int | None = None,
) -> ResultadoSuperficie:
    """Calcula numéricamente el área de la superficie de revolución.

    Args:
        texto_funcion: f(x) escrita por el usuario.
        a, b: Intervalo de integración.
        eje: ``validacion.EJE_X`` o ``validacion.EJE_Y``.
        metodo: ``integracion.SIMPSON`` o ``integracion.TRAPECIO``.
        n: Número de subintervalos.
        area_exacta: Valor exacto conocido (opcional, para los ejemplos).
        n_max_tabla: Mayor n de la tabla de convergencia (por defecto
            max(n, 256), limitado a 2^16).

    Raises:
        SuperficieError: Cualquier error de entrada o matemático, con un
            mensaje pensado para el usuario.
    """
    expr = parsear_funcion(texto_funcion)
    analisis = analizar_funcion(expr, float(a), float(b), eje)
    g = analisis.integrando

    area = FACTOR * integracion.integrar(g, analisis.a, analisis.b, n, metodo)
    ref, err_quad = integracion.referencia_quad(g, analisis.a, analisis.b)
    ref *= FACTOR
    err_quad *= FACTOR

    # Si se conoce el valor exacto, los errores se miden contra él.
    objetivo = area_exacta if area_exacta is not None else ref
    error_abs = abs(area - objetivo)
    error_rel = error_abs / abs(objetivo) if objetivo != 0 else float("inf")

    error_rich = None
    if n >= 4 and (metodo == integracion.TRAPECIO or (n // 2) % 2 == 0):
        mitad = FACTOR * integracion.integrar(g, analisis.a, analisis.b, n // 2, metodo)
        error_rich = integracion.estimacion_richardson(area, mitad, metodo)

    if n_max_tabla is None:
        n_max_tabla = min(max(n, 256), 2**16)
    tabla = integracion.tabla_convergencia(
        g, analisis.a, analisis.b, objetivo, n_max=n_max_tabla, factor=FACTOR
    )

    return ResultadoSuperficie(
        analisis=analisis,
        metodo=metodo,
        n=n,
        area=area,
        area_referencia=ref,
        error_quad=err_quad,
        error_absoluto=error_abs,
        error_relativo=error_rel,
        error_richardson=error_rich,
        convergencia=tabla,
        area_exacta=area_exacta,
    )
