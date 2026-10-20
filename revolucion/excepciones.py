"""Excepciones propias del proyecto.

Todas heredan de ``SuperficieError`` para que la interfaz pueda capturarlas
con un solo ``except`` y mostrar el mensaje al usuario.
"""


class SuperficieError(Exception):
    """Error base del cálculo de superficies de revolución."""


class FuncionInvalidaError(SuperficieError):
    """La función escrita por el usuario no se puede interpretar."""


class IntervaloInvalidoError(SuperficieError):
    """El intervalo [a, b] no es válido (a >= b, valores no finitos, etc.)."""


class DominioError(SuperficieError):
    """La función no está definida en los reales en parte del intervalo."""


class SingularidadError(SuperficieError):
    """La función o el integrando presentan una singularidad no integrable."""


class ParametroInvalidoError(SuperficieError):
    """Parámetro numérico incorrecto (por ejemplo, n impar en Simpson)."""
