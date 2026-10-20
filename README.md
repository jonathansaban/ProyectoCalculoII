# Área de superficies de revolución (cálculo numérico)

Programa en Python que calcula numéricamente el área de la superficie generada
al girar y = f(x), con x en [a, b], alrededor del eje x o del eje y:

- Eje x: S = 2π ∫ₐᵇ |f(x)| √(1 + f'(x)²) dx
- Eje y: S = 2π ∫ₐᵇ |x| √(1 + f'(x)²) dx

La integral se aproxima con **Simpson 1/3** y **trapecio compuesto**
(implementados a mano) y se compara con `scipy.integrate.quad`.

## Estructura

```
proyecto/
├── app.py                  # Interfaz web con Streamlit
├── requirements.txt
├── pytest.ini
├── revolucion/
│   ├── excepciones.py      # Excepciones propias
│   ├── parser.py           # Texto -> SymPy (seguro), derivada, lambdify
│   ├── validacion.py       # Dominio, continuidad, singularidades, integrando
│   ├── integracion.py      # Trapecio, Simpson, quad, Richardson, convergencia
│   ├── calculo.py          # Orquesta todo: calcular_superficie(...)
│   ├── graficas.py         # Plotly: curva 2D, sólido 3D, convergencia
│   └── ejemplos.py         # Esfera, cono, paraboloides con área exacta
└── tests/
    ├── test_parser.py
    ├── test_integracion.py
    └── test_superficies.py
```

## Instalación (solo la primera vez)

Requiere Python 3.10 o superior. Desde la carpeta del proyecto:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si PowerShell no deja activar el entorno, ejecuta una vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Ejecución

Con el entorno activado (`.\.venv\Scripts\Activate.ps1`):

```powershell
python app.py                    # abre la app en http://localhost:8501
python ejercicios_resueltos.py   # compara con ejercicios publicados
pytest                           # ejecuta las pruebas
```

`streamlit run app.py` también funciona. Para cerrar la app, pulsa Ctrl+C en la terminal.

## Cómo escribir f(x)

Se aceptan `2x`, `x^2`, `3sin(x)`, `sqrt(4 - x^2)`, `exp(-x)`, `ln(x)`,
`abs(x)`, `pi`, `e`. La única variable permitida es `x`.

## Tratamiento de singularidades

| Caso | Ejemplo | Comportamiento |
|------|---------|----------------|
| Asíntota vertical | `1/x` en [0, 1] | Error claro |
| Fuera del dominio real | `sqrt(x)` en [-1, 1] | Error claro |
| Salto | `abs(x)/x` en [-1, 1] | Error claro |
| Tangente vertical en un extremo | `sqrt(x)` en 0, esfera en ±r | Aviso; se usa el límite del integrando |
| Integrando divergente | `x^(1/3)` en [0, 1] | Error (integral impropia) |
| Punto anguloso | `abs(x)` en 0 | Aviso; se calcula igual |
| Discontinuidad evitable | `sin(x)/x` en 0 | Aviso; se usa el límite |
