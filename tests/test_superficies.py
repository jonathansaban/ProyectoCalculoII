"""Pruebas del cálculo completo contra áreas exactas conocidas."""

import math

import pytest

from revolucion.calculo import calcular_superficie
from revolucion.ejemplos import EJEMPLOS, area_cono, area_esfera, area_paraboloide
from revolucion.excepciones import (
    DominioError,
    FuncionInvalidaError,
    IntervaloInvalidoError,
    ParametroInvalidoError,
    SingularidadError,
)
from revolucion.integracion import SIMPSON, TRAPECIO
from revolucion.validacion import EJE_X, EJE_Y


@pytest.mark.parametrize("r", [1, 2, 3.5])
def test_esfera(r):
    """La esfera de radio r debe dar 4πr², aunque f' → ∞ en los extremos."""
    res = calcular_superficie(f"sqrt({r}^2 - x^2)", -r, r, EJE_X, SIMPSON, 100)
    assert res.area == pytest.approx(area_esfera(r), rel=1e-10)
    assert res.area_referencia == pytest.approx(area_esfera(r), rel=1e-8)
    assert any("tangente vertical" in aviso for aviso in res.analisis.advertencias)


@pytest.mark.parametrize(("r", "h"), [(3, 4), (1, 1), (2, 5)])
def test_cono(r, h):
    """Superficie lateral del cono: π r √(r² + h²)."""
    res = calcular_superficie(f"{r}x/{h}", 0, h, EJE_X, TRAPECIO, 10)
    assert res.area == pytest.approx(area_cono(r, h), rel=1e-12)


def test_cono_alrededor_del_eje_y():
    """y = x en [0, 1] alrededor del eje y es un cono de r = 1, h = 1."""
    res = calcular_superficie("x", 0, 1, EJE_Y, SIMPSON, 10)
    assert res.area == pytest.approx(area_cono(1, 1), rel=1e-12)


def test_paraboloide_eje_x():
    res = calcular_superficie("sqrt(x)", 0, 2, EJE_X, SIMPSON, 200)
    assert res.area == pytest.approx(area_paraboloide(2), rel=1e-6)


def test_paraboloide_eje_y():
    res = calcular_superficie("x^2", 0, 1, EJE_Y, SIMPSON, 100)
    assert res.area == pytest.approx(math.pi / 6 * (5**1.5 - 1), rel=1e-8)


def test_cilindro():
    """f = 2 en [0, 3]: superficie lateral 2π·2·3 = 12π."""
    res = calcular_superficie("2", 0, 3, EJE_X, SIMPSON, 2)
    assert res.area == pytest.approx(12 * math.pi)


@pytest.mark.parametrize("ejemplo", list(EJEMPLOS.values()), ids=lambda e: e.nombre)
def test_ejemplos_precargados(ejemplo):
    res = calcular_superficie(
        ejemplo.funcion, ejemplo.a, ejemplo.b, ejemplo.eje, SIMPSON, 1000,
        area_exacta=ejemplo.area_exacta,
    )
    assert res.area == pytest.approx(ejemplo.area_exacta, rel=1e-5)
    assert res.area_referencia == pytest.approx(ejemplo.area_exacta, rel=1e-8)


def test_simpson_mas_preciso_que_trapecio():
    simpson = calcular_superficie("exp(x)", 0, 1, EJE_X, SIMPSON, 20)
    trapecio = calcular_superficie("exp(x)", 0, 1, EJE_X, TRAPECIO, 20)
    assert simpson.error_absoluto < trapecio.error_absoluto


def test_estimacion_richardson_es_razonable():
    res = calcular_superficie("exp(x)", 0, 1, EJE_X, TRAPECIO, 64)
    assert res.error_richardson == pytest.approx(res.error_absoluto, rel=0.05)


# ---------------------------------------------------------------- errores


def test_intervalo_invertido():
    with pytest.raises(IntervaloInvalidoError):
        calcular_superficie("x", 2, 1, EJE_X, SIMPSON, 10)


def test_simpson_n_impar():
    with pytest.raises(ParametroInvalidoError):
        calcular_superficie("x", 0, 1, EJE_X, SIMPSON, 11)


def test_funcion_mal_escrita():
    with pytest.raises(FuncionInvalidaError):
        calcular_superficie("x +* 2", 0, 1, EJE_X, SIMPSON, 10)


@pytest.mark.parametrize(("texto", "a", "b"), [("1/x", 0, 1), ("1/x", -1, 1), ("tan(x)", 0, 2)])
def test_asintota_vertical(texto, a, b):
    with pytest.raises(SingularidadError, match="asíntota"):
        calcular_superficie(texto, a, b, EJE_X, SIMPSON, 10)


def test_fuera_del_dominio():
    with pytest.raises(DominioError):
        calcular_superficie("sqrt(x)", -1, 1, EJE_X, SIMPSON, 10)


def test_discontinuidad_de_salto():
    with pytest.raises(SingularidadError, match="discontinua"):
        calcular_superficie("abs(x)/x", -1, 1, EJE_X, SIMPSON, 10)


def test_sqrt_en_cero_avisa_pero_calcula():
    """sqrt(x) en 0: f' → ∞, pero el integrando tiende a 1/2 (integrable)."""
    res = calcular_superficie("sqrt(x)", 0, 1, EJE_X, SIMPSON, 100)
    assert res.analisis.advertencias
    assert res.area == pytest.approx(area_paraboloide(1), rel=1e-6)


def test_integrando_divergente():
    """x^(1/3) en [0, 1]: el integrando ~ x^(-1/3) diverge en 0."""
    with pytest.raises(SingularidadError, match="diverge"):
        calcular_superficie("x^(1/3)", 0, 1, EJE_X, SIMPSON, 10)


def test_punto_anguloso_avisa():
    res = calcular_superficie("abs(x)", -1, 1, EJE_X, SIMPSON, 100)
    assert any("derivable" in aviso for aviso in res.analisis.advertencias)
    # Dos conos de r = 1, h = 1.
    assert res.area == pytest.approx(2 * area_cono(1, 1), rel=1e-10)


def test_discontinuidad_evitable():
    """sin(x)/x tiene límite 1 en 0; se debe poder integrar."""
    res = calcular_superficie("sin(x)/x", -1, 1, EJE_X, SIMPSON, 100)
    assert res.area == pytest.approx(res.area_referencia, rel=1e-8)
