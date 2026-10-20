"""Pruebas del parser seguro de funciones."""

import numpy as np
import pytest
import sympy as sp

from revolucion.excepciones import FuncionInvalidaError
from revolucion.parser import X, a_numerica, derivar, parsear_funcion


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("2x", 2 * X),
        ("x^2", X**2),
        ("3x^2 + 2x + 1", 3 * X**2 + 2 * X + 1),
        ("sqrt(4 - x^2)", sp.sqrt(4 - X**2)),
        ("2sin(x)", 2 * sp.sin(X)),
        ("ln(x)", sp.log(X)),
        ("e^x", sp.exp(X)),
        ("pi*x", sp.pi * X),
        ("5", sp.Integer(5)),
    ],
)
def test_parseo_valido(texto, esperado):
    assert sp.simplify(parsear_funcion(texto) - esperado) == 0


@pytest.mark.parametrize(
    "texto",
    [
        "",
        "   ",
        "x +",
        "(x + 1",
        "y + 1",
        "__import__('os')",
        "x; print(1)",
        "open(x)",
        "lambda x: x",
        "x" * 300,
    ],
)
def test_parseo_invalido(texto):
    with pytest.raises(FuncionInvalidaError):
        parsear_funcion(texto)


def test_derivada_simbolica():
    expr = parsear_funcion("x^3 + sin(x)")
    assert sp.simplify(derivar(expr) - (3 * X**2 + sp.cos(X))) == 0


def test_numerica_constante_devuelve_arreglo():
    f = a_numerica(parsear_funcion("7"))
    valores = f(np.array([0.0, 1.0, 2.0]))
    assert valores.shape == (3,)
    assert np.allclose(valores, 7)


def test_numerica_fuera_de_dominio_da_nan():
    f = a_numerica(parsear_funcion("sqrt(x)"))
    assert np.isnan(f(np.array([-1.0]))[0])
