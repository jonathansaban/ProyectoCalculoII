"""Pruebas de los métodos numéricos implementados a mano."""

import math

import numpy as np
import pytest

from revolucion import integracion
from revolucion.excepciones import ParametroInvalidoError


def test_trapecio_exacto_en_rectas():
    assert integracion.trapecio(lambda x: 3 * x + 1, 0, 2, 1) == pytest.approx(8)


def test_simpson_exacto_en_cubicas():
    resultado = integracion.simpson(lambda x: x**3 - 2 * x, 0, 2, 2)
    assert resultado == pytest.approx(0, abs=1e-14)


def test_simpson_seno():
    assert integracion.simpson(np.sin, 0, math.pi, 100) == pytest.approx(2, rel=1e-8)


def test_simpson_n_impar_falla():
    with pytest.raises(ParametroInvalidoError, match="par"):
        integracion.simpson(np.sin, 0, 1, 7)


@pytest.mark.parametrize("n", [0, -4, 2.5])
def test_n_invalido(n):
    with pytest.raises(ParametroInvalidoError):
        integracion.trapecio(np.sin, 0, 1, n)


@pytest.mark.parametrize(
    ("metodo", "orden"), [(integracion.TRAPECIO, 2), (integracion.SIMPSON, 4)]
)
def test_orden_de_convergencia(metodo, orden):
    """El orden observado debe coincidir con el teórico para una función suave."""
    filas = integracion.tabla_convergencia(np.exp, 0, 1, math.e - 1, n_max=64)
    ordenes = [
        f.orden_trapecio if metodo == integracion.TRAPECIO else f.orden_simpson
        for f in filas[2:]
    ]
    assert all(o == pytest.approx(orden, abs=0.1) for o in ordenes if o is not None)


def test_referencia_quad():
    valor, error = integracion.referencia_quad(np.cos, 0, math.pi / 2)
    assert valor == pytest.approx(1)
    assert error < 1e-10
