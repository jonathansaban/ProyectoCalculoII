"""Interpretación segura de funciones escritas como texto.

Convierte cadenas como ``"2x^2 + sin(x)"`` en expresiones de SymPy, calcula
la derivada simbólica y genera funciones numéricas vectorizadas con NumPy.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from tokenize import TokenError

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from revolucion.excepciones import FuncionInvalidaError

# Variable independiente. Se declara real para que SymPy simplifique
# correctamente expresiones como Abs(x) o sqrt(x**2).
X = sp.Symbol("x", real=True)

LONGITUD_MAXIMA = 200

# Solo se permiten dígitos, letras, espacios, operadores y paréntesis.
# Esto bloquea comillas, corchetes, guiones bajos dobles, etc., y evita
# que parse_expr (que internamente usa eval) ejecute código arbitrario.
_PATRON_PERMITIDO = re.compile(r"^[0-9a-zA-Z\s\.\+\-\*/\^\(\),]+$")

# Nombres que el usuario puede escribir. Cualquier otro nombre se rechaza.
NOMBRES_PERMITIDOS: dict[str, object] = {
    "x": X,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}

# parse_expr necesita estos nombres de SymPy para construir la expresión
# (Integer, Float, Symbol...). No se exponen funciones de Python.
_GLOBALES_SYMPY: dict[str, object] = {
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
}

_TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

FuncionNumerica = Callable[[np.ndarray], np.ndarray]


def parsear_funcion(texto: str) -> sp.Expr:
    """Convierte el texto del usuario en una expresión de SymPy en ``x``.

    Acepta multiplicación implícita (``2x``, ``3sin(x)``) y ``^`` como
    potencia.

    Args:
        texto: Función escrita por el usuario, por ejemplo ``"sqrt(4 - x^2)"``.

    Returns:
        Expresión simbólica que depende (como mucho) de ``x``.

    Raises:
        FuncionInvalidaError: Si el texto está vacío, contiene caracteres o
            nombres no permitidos, o tiene errores de sintaxis.
    """
    texto = (texto or "").strip()
    if not texto:
        raise FuncionInvalidaError("Escribe una función f(x).")
    if len(texto) > LONGITUD_MAXIMA:
        raise FuncionInvalidaError(
            f"La función es demasiado larga (máximo {LONGITUD_MAXIMA} caracteres)."
        )
    if not _PATRON_PERMITIDO.match(texto):
        raise FuncionInvalidaError(
            "La función contiene caracteres no permitidos. Usa solo números, "
            "x, operadores (+ - * / ^), paréntesis y funciones como sin, "
            "cos, exp, log, sqrt."
        )

    # Se valida cada palabra antes de evaluar nada.
    for nombre in re.findall(r"[a-zA-Z]+", texto):
        if nombre not in NOMBRES_PERMITIDOS:
            raise FuncionInvalidaError(
                f"Nombre desconocido: '{nombre}'. La variable debe ser x y "
                "las funciones permitidas son: "
                + ", ".join(sorted(k for k in NOMBRES_PERMITIDOS if len(k) > 1))
            )

    try:
        expr = parse_expr(
            texto,
            local_dict=dict(NOMBRES_PERMITIDOS),
            global_dict={"__builtins__": {}, **_GLOBALES_SYMPY},
            transformations=_TRANSFORMACIONES,
            evaluate=True,
        )
    except (
        SyntaxError,
        TokenError,
        TypeError,
        ValueError,
        AttributeError,
        ZeroDivisionError,
        sp.SympifyError,
    ) as exc:
        raise FuncionInvalidaError(
            f"No se pudo interpretar la función '{texto}'. Revisa paréntesis "
            "y operadores."
        ) from exc

    if not isinstance(expr, sp.Expr):
        raise FuncionInvalidaError("La entrada no es una expresión matemática.")

    if expr.free_symbols - {X}:
        raise FuncionInvalidaError("La función solo puede depender de la variable x.")

    if expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan, sp.I):
        raise FuncionInvalidaError(
            "La función contiene valores infinitos, indefinidos o complejos."
        )

    return expr


def derivar(expr: sp.Expr) -> sp.Expr:
    """Devuelve la derivada simbólica ``f'(x)``."""
    return sp.diff(expr, X)


def a_numerica(expr: sp.Expr) -> FuncionNumerica:
    """Convierte una expresión de SymPy en una función vectorizada de NumPy.

    La función devuelta siempre retorna un arreglo del mismo tamaño que la
    entrada (incluso para expresiones constantes) y produce ``nan`` en lugar
    de lanzar advertencias cuando la expresión no está definida.
    """
    funcion = sp.lambdify(X, expr, modules="numpy")

    def evaluar(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        with np.errstate(all="ignore"):
            valores = funcion(xs)
        valores = np.asarray(valores)
        if np.iscomplexobj(valores):
            # Se descartan resultados con parte imaginaria apreciable.
            reales = np.where(np.abs(valores.imag) < 1e-12, valores.real, np.nan)
            valores = reales
        return np.broadcast_to(valores.astype(float), xs.shape).copy()

    return evaluar
