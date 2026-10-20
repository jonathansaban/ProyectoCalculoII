"""Ejemplos precargados con su área exacta conocida."""

from __future__ import annotations

import math
from dataclasses import dataclass

from revolucion.validacion import EJE_X, EJE_Y


@dataclass(frozen=True)
class Ejemplo:
    """Caso de prueba con solución analítica."""

    nombre: str
    funcion: str
    a: float
    b: float
    eje: str
    area_exacta: float
    formula: str


def area_esfera(r: float) -> float:
    """Esfera de radio r: S = 4πr²."""
    return 4 * math.pi * r**2


def area_cono(r: float, h: float) -> float:
    """Superficie lateral de un cono de radio r y altura h: S = πr·√(r² + h²)."""
    return math.pi * r * math.sqrt(r**2 + h**2)


def area_paraboloide(h: float) -> float:
    """f(x) = √x en [0, h] alrededor del eje x: S = (π/6)·((4h + 1)^(3/2) − 1)."""
    return math.pi / 6 * ((4 * h + 1) ** 1.5 - 1)


EJEMPLOS: dict[str, Ejemplo] = {
    e.nombre: e
    for e in (
        Ejemplo(
            "Esfera (r = 2)",
            "sqrt(4 - x^2)",
            -2.0,
            2.0,
            EJE_X,
            area_esfera(2),
            "S = 4πr² = 16π",
        ),
        Ejemplo(
            "Cono (r = 3, h = 4)",
            "3x/4",
            0.0,
            4.0,
            EJE_X,
            area_cono(3, 4),
            "S = πr·√(r² + h²) = 15π",
        ),
        Ejemplo(
            "Paraboloide (√x, h = 2)",
            "sqrt(x)",
            0.0,
            2.0,
            EJE_X,
            area_paraboloide(2),
            "S = (π/6)·((4h + 1)^(3/2) − 1)",
        ),
        Ejemplo(
            "Paraboloide (x², eje y)",
            "x^2",
            0.0,
            1.0,
            EJE_Y,
            math.pi / 6 * (5**1.5 - 1),
            "S = (π/6)·(5^(3/2) − 1)",
        ),
    )
}
