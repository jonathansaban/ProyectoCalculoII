"""Validación matemática de f en [a, b] y construcción del integrando.

Además de rechazar funciones no definidas, discontinuas o con asíntotas,
este módulo trata los casos "salvables": por ejemplo la esfera
f(x) = sqrt(r² - x²), donde f'(x) → ∞ en los extremos pero el integrando
|f| * sqrt(1 + f'²) tiende a r. En esos puntos se usa el límite del
integrando en lugar de evaluarlo directamente (que daría 0 * ∞ = nan).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from revolucion.excepciones import (
    DominioError,
    IntervaloInvalidoError,
    SingularidadError,
)
from revolucion.parser import X, FuncionNumerica, a_numerica, derivar

EJE_X = "Eje x"
EJE_Y = "Eje y"
EJES = (EJE_X, EJE_Y)

# Número de puntos de la verificación numérica de respaldo.
PUNTOS_MUESTREO = 4001


@dataclass
class FuncionAnalizada:
    """Resultado de analizar f en [a, b] para un eje de rotación dado."""

    expr: sp.Expr
    derivada: sp.Expr
    integrando_expr: sp.Expr
    f: FuncionNumerica
    df: FuncionNumerica
    integrando: FuncionNumerica
    a: float
    b: float
    eje: str
    advertencias: list[str] = field(default_factory=list)


def validar_intervalo(a: float, b: float) -> None:
    """Comprueba que a y b sean finitos y que a < b."""
    if not (np.isfinite(a) and np.isfinite(b)):
        raise IntervaloInvalidoError("Los extremos a y b deben ser números finitos.")
    if a >= b:
        raise IntervaloInvalidoError(f"Debe cumplirse a < b (se recibió a = {a}, b = {b}).")


def construir_integrando(expr: sp.Expr, derivada: sp.Expr, eje: str) -> sp.Expr:
    """Integrando sin el factor 2π.

    - Eje x: |f(x)| * sqrt(1 + f'(x)²)
    - Eje y: |x|    * sqrt(1 + f'(x)²)
    """
    longitud = sp.sqrt(1 + derivada**2)
    if eje == EJE_X:
        return sp.Abs(expr) * longitud
    if eje == EJE_Y:
        return sp.Abs(X) * longitud
    raise ValueError(f"Eje desconocido: {eje}")


def _exacto(valor: float) -> sp.Rational:
    """Convierte un float a racional exacto para que SymPy calcule límites fiables."""
    return sp.Rational(repr(float(valor)))


def _limite(expr: sp.Expr, punto: sp.Expr, direccion: str) -> sp.Expr | None:
    """Límite lateral; devuelve None si SymPy no lo puede calcular."""
    try:
        resultado = sp.limit(expr, X, punto, dir=direccion)
    except (NotImplementedError, ValueError, TypeError):
        return None
    if resultado.has(sp.AccumBounds) or resultado.has(sp.Limit):
        return None
    return resultado


def _es_finito(valor: sp.Expr | None) -> bool:
    return valor is not None and bool(valor.is_finite) and valor.is_real is not False


def _valor_directo_finito(expr: sp.Expr, punto: sp.Expr) -> bool:
    """Indica si ``expr`` evaluada en ``punto`` da un número real finito."""
    try:
        valor = complex(expr.subs(X, punto).evalf())
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return bool(np.isfinite(valor)) and abs(valor.imag) < 1e-12


def _puntos_problematicos(
    expr: sp.Expr, intervalo: sp.Interval, nombre: str
) -> set[sp.Expr] | None:
    """Puntos de [a, b] donde ``expr`` no es continua.

    Returns:
        Conjunto (posiblemente vacío) de puntos aislados, o None si SymPy no
        pudo determinar el dominio de continuidad.

    Raises:
        DominioError: Si la expresión no está definida en un subintervalo.
    """
    try:
        dominio = sp.calculus.util.continuous_domain(expr, X, intervalo)
    except (NotImplementedError, ValueError, TypeError):
        return None

    excluido = sp.Complement(intervalo, dominio)
    if excluido is sp.S.EmptySet:
        return set()
    if isinstance(excluido, sp.FiniteSet):
        return set(excluido)
    if isinstance(excluido, (sp.Interval, sp.Union)):
        # Si lo excluido tiene longitud positiva, f no está definida ahí.
        if excluido.measure != 0:
            raise DominioError(
                f"{nombre} no está definida en los números reales en "
                f"{_formatear_conjunto(excluido)}. Ajusta el intervalo [a, b]."
            )
        return set(excluido.boundary)
    return None


def _puntos_angulosos(expr: sp.Expr, intervalo: sp.Interval) -> set[sp.Expr]:
    """Ceros (en [a, b]) de los argumentos de Abs, sign y Heaviside.

    SymPy considera continua a sign(x), así que continuous_domain no detecta
    que |x| no es derivable en 0; se buscan esos puntos explícitamente.
    """
    puntos: set[sp.Expr] = set()
    for nodo in expr.atoms(sp.Abs, sp.sign, sp.Heaviside):
        try:
            ceros = sp.solveset(nodo.args[0], X, intervalo)
        except (NotImplementedError, ValueError, TypeError):
            continue
        if isinstance(ceros, sp.FiniteSet):
            puntos |= set(ceros)
    return puntos


def _formatear_conjunto(conjunto: sp.Set) -> str:
    try:
        return str(conjunto.evalf(4))
    except (TypeError, AttributeError):
        return str(conjunto)


def analizar_funcion(expr: sp.Expr, a: float, b: float, eje: str) -> FuncionAnalizada:
    """Valida f en [a, b] y devuelve el integrando listo para integrar.

    Raises:
        IntervaloInvalidoError, DominioError, SingularidadError.
    """
    validar_intervalo(a, b)
    if eje not in EJES:
        raise ValueError(f"Eje desconocido: {eje}")

    derivada = derivar(expr)
    integrando_expr = construir_integrando(expr, derivada, eje)
    advertencias: list[str] = []

    a_ex, b_ex = _exacto(a), _exacto(b)
    intervalo = sp.Interval(a_ex, b_ex)

    # 1) Análisis simbólico de continuidad de f y de f'.
    puntos_f = _puntos_problematicos(expr, intervalo, "f(x)")
    puntos_df = _puntos_problematicos(derivada, intervalo, "f'(x)")
    analisis_simbolico = puntos_f is not None and puntos_df is not None
    if not analisis_simbolico:
        advertencias.append(
            "No se pudo analizar simbólicamente la continuidad; se usa solo la "
            "verificación numérica."
        )
    puntos_f = puntos_f or set()
    puntos_df = (puntos_df or set()) | _puntos_angulosos(expr, intervalo)

    # 2) Estudio de cada punto especial (incluidos siempre los extremos).
    especiales: dict[float, float] = {}
    candidatos = sorted(puntos_f | puntos_df | {a_ex, b_ex}, key=float)
    for p in candidatos:
        valor = _estudiar_punto(
            expr, integrando_expr, p, a_ex, b_ex, p in puntos_f, p in puntos_df, advertencias
        )
        if valor is not None:
            especiales[float(p)] = valor

    integrando = _integrando_numerico(integrando_expr, especiales, a, b)
    f_num = a_numerica(expr)
    df_num = a_numerica(derivada)

    # 3) Verificación numérica de respaldo sobre una malla fina.
    _verificar_numericamente(f_num, integrando, a, b, especiales)

    if eje == EJE_Y and a < 0 < b:
        advertencias.append(
            "El intervalo contiene x = 0 y se rota alrededor del eje y: las "
            "partes con x < 0 y x > 0 generan superficies que se superponen y "
            "ambas se suman al área."
        )

    return FuncionAnalizada(
        expr=expr,
        derivada=derivada,
        integrando_expr=integrando_expr,
        f=f_num,
        df=df_num,
        integrando=integrando,
        a=a,
        b=b,
        eje=eje,
        advertencias=advertencias,
    )


def _estudiar_punto(
    expr: sp.Expr,
    integrando_expr: sp.Expr,
    p: sp.Expr,
    a: sp.Expr,
    b: sp.Expr,
    problema_en_f: bool,
    problema_en_df: bool,
    advertencias: list[str],
) -> float | None:
    """Analiza un punto especial y devuelve el valor a usar para el integrando.

    Devuelve None si el integrando se puede evaluar directamente en ``p``.
    """
    p_txt = f"{float(p):.6g}"
    lados = [d for d, activo in (("-", p > a), ("+", p < b)) if activo]

    f_ok = _valor_directo_finito(expr, p)
    directo_ok = _valor_directo_finito(integrando_expr, p)
    if not problema_en_f and not problema_en_df and f_ok and directo_ok:
        return None

    # f debe tener límite finito por cada lado (sin asíntotas verticales).
    if problema_en_f or not f_ok:
        limites_f = [_limite(expr, p, d) for d in lados]
        if any(v is not None and not _es_finito(v) for v in limites_f):
            raise SingularidadError(
                f"f(x) tiene una asíntota vertical (singularidad) en x = {p_txt}. "
                "La superficie no es finita o el método no puede evaluarla; "
                "excluye ese punto del intervalo."
            )
        if len(limites_f) == 2 and None not in limites_f:
            if sp.simplify(limites_f[0] - limites_f[1]) != 0:
                raise SingularidadError(
                    f"f(x) es discontinua en x = {p_txt} (salto de "
                    f"{float(limites_f[0]):.4g} a {float(limites_f[1]):.4g})."
                )

    limites_g = [_limite(integrando_expr, p, d) for d in lados]
    if any(v is None for v in limites_g):
        if directo_ok:
            return None
        raise SingularidadError(
            f"No se pudo determinar el comportamiento del integrando en x = {p_txt}."
        )
    if any(not _es_finito(v) for v in limites_g):
        raise SingularidadError(
            f"El integrando diverge en x = {p_txt} (f'(x) → ∞ demasiado rápido). "
            "Es una integral impropia que Simpson y el trapecio no pueden "
            "evaluar en ese nodo; ajusta el intervalo."
        )

    valor = float(sum(float(v) for v in limites_g) / len(limites_g))
    if problema_en_f or not f_ok:
        advertencias.append(
            f"f(x) tiene una discontinuidad evitable en x = {p_txt}; se usa su límite."
        )
    elif problema_en_df or not directo_ok:
        if p in (a, b):
            advertencias.append(
                f"f'(x) no está definida en el extremo x = {p_txt} (tangente "
                f"vertical). Se usa el límite del integrando ({valor:.6g})."
            )
        else:
            advertencias.append(
                f"f(x) no es derivable en x = {p_txt} (punto anguloso o "
                "tangente vertical). El integrando sigue siendo integrable, "
                "pero la convergencia puede ser más lenta."
            )
    return valor


def _integrando_numerico(
    integrando_expr: sp.Expr, especiales: dict[float, float], a: float, b: float
) -> FuncionNumerica:
    """Integrando vectorizado que sustituye los puntos especiales por su límite."""
    base = a_numerica(integrando_expr)
    puntos = np.array(list(especiales.keys()), dtype=float)
    valores = np.array(list(especiales.values()), dtype=float)
    tolerancia = 1e-12 * max(1.0, b - a)

    def g(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        y = base(xs)
        for punto, valor in zip(puntos, valores):
            y[np.abs(xs - punto) <= tolerancia] = valor
        return y

    return g


def _verificar_numericamente(
    f: FuncionNumerica,
    integrando: FuncionNumerica,
    a: float,
    b: float,
    especiales: dict[float, float],
) -> None:
    """Evalúa f y el integrando en una malla fina y detecta valores inválidos.

    Los puntos especiales ya estudiados (con límite finito) se excluyen.
    """
    malla = np.linspace(a, b, PUNTOS_MUESTREO)
    tolerancia = 1e-12 * max(1.0, b - a)
    excusados = np.zeros(malla.shape, dtype=bool)
    for punto in especiales:
        excusados |= np.abs(malla - punto) <= tolerancia
    for nombre, funcion in (("f(x)", f), ("el integrando", integrando)):
        valores = funcion(malla)
        malos = ~np.isfinite(valores) & ~excusados
        if malos.any():
            x_malo = malla[np.argmax(malos)]
            raise SingularidadError(
                f"{nombre} no es finito o no está definido en x ≈ {x_malo:.6g}. "
                "Revisa el dominio de la función en [a, b]."
            )
