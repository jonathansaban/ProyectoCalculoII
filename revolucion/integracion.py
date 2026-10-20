"""Métodos de integración numérica.

Simpson 1/3 y trapecio compuesto están implementados a mano. SciPy solo se
usa como valor de referencia (``scipy.integrate.quad``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from revolucion.excepciones import ParametroInvalidoError

FuncionNumerica = Callable[[np.ndarray], np.ndarray]

TRAPECIO = "Trapecio compuesto"
SIMPSON = "Simpson 1/3"
METODOS = (SIMPSON, TRAPECIO)

# Orden de convergencia teórico de cada método (error ~ h^p).
ORDEN_TEORICO = {TRAPECIO: 2, SIMPSON: 4}


def _validar_n(n: int, minimo: int = 1) -> None:
    """Comprueba que n sea un entero positivo razonable."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise ParametroInvalidoError("n debe ser un número entero.")
    if n < minimo:
        raise ParametroInvalidoError(f"n debe ser mayor o igual que {minimo}.")
    if n > 10_000_000:
        raise ParametroInvalidoError("n es demasiado grande (máximo 10 000 000).")


def trapecio(g: FuncionNumerica, a: float, b: float, n: int) -> float:
    """Regla del trapecio compuesta con ``n`` subintervalos.

    T_n = h * [ g(x0)/2 + g(x1) + ... + g(x_{n-1}) + g(xn)/2 ],  h = (b-a)/n
    """
    _validar_n(n)
    h = (b - a) / n
    nodos = np.linspace(a, b, n + 1)
    y = g(nodos)
    return float(h * (y.sum() - 0.5 * (y[0] + y[-1])))


def simpson(g: FuncionNumerica, a: float, b: float, n: int) -> float:
    """Regla de Simpson 1/3 compuesta con ``n`` subintervalos (n par).

    S_n = h/3 * [ g(x0) + 4*sum(impares) + 2*sum(pares interiores) + g(xn) ]
    """
    _validar_n(n, minimo=2)
    if n % 2 != 0:
        raise ParametroInvalidoError(
            f"Simpson 1/3 necesita un número par de subintervalos (n = {n} es impar)."
        )
    h = (b - a) / n
    nodos = np.linspace(a, b, n + 1)
    y = g(nodos)
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def integrar(g: FuncionNumerica, a: float, b: float, n: int, metodo: str) -> float:
    """Integra ``g`` en [a, b] con el método indicado."""
    if metodo == SIMPSON:
        return simpson(g, a, b, n)
    if metodo == TRAPECIO:
        return trapecio(g, a, b, n)
    raise ParametroInvalidoError(f"Método desconocido: {metodo}")


def referencia_quad(g: FuncionNumerica, a: float, b: float) -> tuple[float, float]:
    """Valor de referencia con ``scipy.integrate.quad`` (Gauss-Kronrod adaptativo).

    Returns:
        Tupla ``(valor, error_estimado_por_quad)``.
    """

    def escalar(t: float) -> float:
        return float(g(np.array([t]))[0])

    valor, error = integrate.quad(escalar, a, b, limit=500)
    return float(valor), float(error)


def estimacion_richardson(valor_n: float, valor_n_medios: float, metodo: str) -> float:
    """Estima el error de I_n comparándolo con I_{n/2} (extrapolación de Richardson).

    Para un método de orden p:  error(I_n) ≈ |I_n - I_{n/2}| / (2^p - 1).
    """
    p = ORDEN_TEORICO[metodo]
    return abs(valor_n - valor_n_medios) / (2**p - 1)


@dataclass(frozen=True)
class FilaConvergencia:
    """Una fila de la tabla de convergencia."""

    n: int
    trapecio: float
    error_trapecio: float
    orden_trapecio: float | None
    simpson: float
    error_simpson: float
    orden_simpson: float | None


def tabla_convergencia(
    g: FuncionNumerica,
    a: float,
    b: float,
    valor_referencia: float,
    n_max: int = 1024,
    factor: float = 1.0,
) -> list[FilaConvergencia]:
    """Calcula ambos métodos para n = 2, 4, 8, ..., n_max.

    El "orden observado" es log2(error(n/2) / error(n)); debería acercarse a
    2 en el trapecio y a 4 en Simpson cuando la función es suave.

    Args:
        g: Integrando.
        a, b: Extremos del intervalo.
        valor_referencia: Valor "exacto" con el que se miden los errores
            (ya multiplicado por ``factor``).
        n_max: Mayor n de la tabla.
        factor: Constante que multiplica la integral (2π en superficies).
    """
    filas: list[FilaConvergencia] = []
    n = 2
    anterior_t = anterior_s = None
    while n <= max(n_max, 2):
        t = factor * trapecio(g, a, b, n)
        s = factor * simpson(g, a, b, n)
        err_t = abs(t - valor_referencia)
        err_s = abs(s - valor_referencia)
        filas.append(
            FilaConvergencia(
                n=n,
                trapecio=t,
                error_trapecio=err_t,
                orden_trapecio=_orden(anterior_t, err_t),
                simpson=s,
                error_simpson=err_s,
                orden_simpson=_orden(anterior_s, err_s),
            )
        )
        anterior_t, anterior_s = err_t, err_s
        n *= 2
    return filas


def _orden(error_anterior: float | None, error_actual: float) -> float | None:
    """Orden observado entre dos filas consecutivas (None si no es calculable)."""
    # Por debajo de este umbral el error es ruido de redondeo.
    umbral = 1e-13
    if error_anterior is None or error_anterior < umbral or error_actual < umbral:
        return None
    return float(np.log2(error_anterior / error_actual))
