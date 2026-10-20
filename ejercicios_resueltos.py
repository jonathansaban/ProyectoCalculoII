"""Compara el programa con ejercicios resueltos publicados en internet.

Ejecutar con:  python ejercicios_resueltos.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from revolucion.calculo import calcular_superficie
from revolucion.integracion import SIMPSON
from revolucion.validacion import EJE_X, EJE_Y

PI = math.pi
R2 = math.sqrt(2)


@dataclass(frozen=True)
class Ejercicio:
    fuente: str
    enunciado: str
    funcion: str
    a: float
    b: float
    eje: str
    respuesta_texto: str
    respuesta: float


PAUL = "Paul's Online Notes (Lamar Univ.)"
LIBRE = "OpenStax / LibreTexts, sec. 2.4"
YALE = "Yale Calculus Tutorials"

EJERCICIOS = [
    Ejercicio(PAUL, "y = sqrt(9 - x^2), -2 <= x <= 2, eje x",
              "sqrt(9 - x^2)", -2, 2, EJE_X, "24π", 24 * PI),
    Ejercicio(PAUL, "y = raíz cúbica de x, 1 <= y <= 2, eje y  (=> 1 <= x <= 8)",
              "x^(1/3)", 1, 8, EJE_Y, "π/27·(145^(3/2) − 10^(3/2)) ≈ 199.48",
              PI / 27 * (145**1.5 - 10**1.5)),
    Ejercicio("Math StackExchange #1925496", "y = x^3, 0 <= x <= 1, eje x",
              "x^3", 0, 1, EJE_X, "π/27·(10√10 − 1)", PI / 27 * (10 * math.sqrt(10) - 1)),
    Ejercicio(YALE, "y = sin(x), 0 <= x <= π, eje x",
              "sin(x)", 0, PI, EJE_X, "2π(√2 + ½ln(1+√2) − ½ln(√2−1))",
              2 * PI * (R2 + 0.5 * math.log(1 + R2) - 0.5 * math.log(R2 - 1))),
    # Yale publica (5√5 − 1)/12 ≈ 0.848: es una errata (le falta el π y el
    # denominador es 6). El valor correcto se obtiene integrando a mano.
    Ejercicio(YALE, "y = sqrt(x), 0 <= x <= 1, eje x  (corrigiendo errata de la página)",
              "sqrt(x)", 0, 1, EJE_X, "π(5√5 − 1)/6", PI * (5 * math.sqrt(5) - 1) / 6),
    Ejercicio(YALE, "y = sqrt(x + 1), 0 <= x <= 1, eje x",
              "sqrt(x + 1)", 0, 1, EJE_X, "π(27 − 5√5)/6", PI * (27 - 5 * math.sqrt(5)) / 6),
    Ejercicio(YALE, "y = sqrt(2x - x^2), 0 <= x <= 1, eje x",
              "sqrt(2x - x^2)", 0, 1, EJE_X, "2π", 2 * PI),
    Ejercicio(LIBRE, "Ej. 27: y = sqrt(x), de x = 2 a x = 6, eje x",
              "sqrt(x)", 2, 6, EJE_X, "49π/3", 49 * PI / 3),
    Ejercicio(LIBRE, "Ej. 29: y = 7x, de x = -1 a x = 1, eje x",
              "7x", -1, 1, EJE_X, "70π√2", 70 * PI * R2),
    Ejercicio(LIBRE, "Ej. 31: y = sqrt(4 - x^2), de x = 0 a x = 2, eje x",
              "sqrt(4 - x^2)", 0, 2, EJE_X, "8π", 8 * PI),
    Ejercicio(LIBRE, "Ej. 33: y = 5x, de x = 1 a x = 5, eje x",
              "5x", 1, 5, EJE_X, "120π√26", 120 * PI * math.sqrt(26)),
    Ejercicio(LIBRE, "Ej. 35: y = x^2, de x = 0 a x = 2, eje y",
              "x^2", 0, 2, EJE_Y, "π/6·(17√17 − 1)", PI / 6 * (17 * math.sqrt(17) - 1)),
    Ejercicio(LIBRE, "Ej. 37: y = x + 1, de x = 0 a x = 3, eje y",
              "x + 1", 0, 3, EJE_Y, "9√2·π", 9 * R2 * PI),
    Ejercicio(LIBRE, "Ej. 39: y = raíz cúbica de x, de x = 1 a x = 27, eje y",
              "x^(1/3)", 1, 27, EJE_Y, "10√10·π/27·(73√73 − 1)  [como aparece publicado]",
              10 * math.sqrt(10) * PI / 27 * (73 * math.sqrt(73) - 1)),
    Ejercicio(LIBRE, "Ej. 41: y = 1/sqrt(x), de x = 1 a x = 3, eje y",
              "1/sqrt(x)", 1, 3, EJE_Y, "≈ 25.645", 25.645),
]


def main() -> None:
    # La consola de Windows no usa UTF-8 por defecto (π, √, ≈).
    sys.stdout.reconfigure(encoding="utf-8")
    for i, ej in enumerate(EJERCICIOS, 1):
        res = calcular_superficie(ej.funcion, ej.a, ej.b, ej.eje, SIMPSON, 1000)
        dif = abs(res.area - ej.respuesta) / abs(ej.respuesta)
        veredicto = "COINCIDE" if dif < 1e-3 else "NO COINCIDE"
        print(f"{i:2d}. [{ej.fuente}] {ej.enunciado}")
        print(f"    Entrada: f(x) = {ej.funcion}, a = {ej.a:g}, b = {ej.b:g}, {ej.eje}")
        print(f"    Publicado: {ej.respuesta_texto} = {ej.respuesta:.6f}")
        print(f"    Programa (Simpson, n=1000): {res.area:.6f}   "
              f"quad: {res.area_referencia:.6f}   -> {veredicto} "
              f"(dif. relativa {dif:.1e})\n")


if __name__ == "__main__":
    main()
