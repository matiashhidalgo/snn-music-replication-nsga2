# Réplica de Liang, Zeng & Tang (2026) + NSGA-II de hiperparámetros

Proyecto de **Computación Bioinspirada** (Tec de Monterrey, Guadalajara), parte 2. Réplica de un artículo con un algoritmo bioinspirado, una mejora propuesta y su evaluación estadística.

> **Artículo replicado:** Liang, Zeng & Tang (2026). *A spiking neural network inspired by neuroscience and psychology for Western mode- and key-conditioned music learning and composition.* Scientific Reports 16:12956. https://doi.org/10.1038/s41598-026-43529-1

**Si vas a ejecutar el código, ve directo a [`docs/GUIA_EJECUCION.md`](docs/GUIA_EJECUCION.md).** Está escrita paso a paso, sin necesidad de entender el código.

---

## De qué trata

El artículo presenta una red neuronal de pulsos (*spiking neural network*, SNN) con neuronas de Izhikevich y plasticidad STDP que aprende corales de Bach y genera piezas en una tonalidad dada. Su código es público (BrainCog), pero **el artículo no define una función objetivo ni una restricción**, y no da el valor de varios hiperparámetros.

Este repositorio hace tres cosas:

1. **Replica** los resultados del artículo con el código del autor: alineación de la red con el perfil psicológico de Krumhansl–Schmuckler (Fig. 5e), métricas de las piezas generadas (Fig. 7b, Tabla 2) y distancia euclidiana estandarizada, SED (Fig. 8a). Configuración **C0** = parámetros por defecto del código, **30 semillas independientes**.
2. **Propone una mejora:** un algoritmo genético multiobjetivo (**NSGA-II**, `pymoo`) que ajusta 6 hiperparámetros de la SNN.
   - Objetivo 1: SED entre piezas generadas y piezas reales (Ec. 16 del artículo).
   - Objetivo 2: 1 − coseno entre los pesos sinápticos de la red y el perfil KS.
   - Restricción: tasa de notas diatónicas (DPR) ≥ 0.90.
   - La configuración elegida del frente de Pareto es **C1**, también con **30 semillas** que el algoritmo nunca vio.
3. **Compara estadísticamente** réplica, artículo y mejora (Wilcoxon, corrección de Holm) y documenta qué tan replicable fue el artículo.

Hallazgos del análisis del código (detalle en `docs/PLAN_TECNICO.md`, sección 2): el perfil KS **no** se inyecta en el modelo, y la "alineación con KS" que reporta el artículo se parece a un histograma de alturas. Por eso el análisis incluye un control sin SNN (histograma vs KS). También hay discrepancias entre el artículo y el código (`d` de Izhikevich 30 vs 8, umbral de sinapsis 5 vs 2).

## Qué se espera de la ejecución

| Fase | Qué hace | Duración | Resultado |
|---|---|---|---|
| **A** | Instala, comprueba el Mac, prueba el flujo con la SNN real (1 coral) y mide tiempos | ≈ 30–40 min | `FASE A OK` y el informe `results_smoke/informe_prueba.md` |
| **B** | 30 corridas de C0 → algoritmo genético → verificación → 30 corridas de C1 → análisis | ≈ 28 h, sin atención | `TODO COMPLETO` y `results_bundle.zip` |

Todo es **reanudable**: si se interrumpe, se relanza la misma orden y continúa. Todo se guarda con datos crudos, así que el análisis se rehace sin volver a correr la SNN.

**Qué hay dentro de `results_bundle.zip`** (lo único que se manda de vuelta):

- Datos crudos: 30 corridas de C0 y de C1, historial completo del AG, frente de Pareto, caché de evaluaciones.
- Tablas T1–T5 (réplica vs artículo, estadística por configuración, pruebas de Wilcoxon, parámetros, replicabilidad y dificultades).
- Figuras F1–F6 (convergencia, frente de Pareto, PSC/PASW vs KS, cajas, control del histograma, métricas).
- `manifest.json`, `bench.json` y los registros (`logs/`).

**Qué resultados son razonables** (no hay garantía de que salgan así; cualquiera de los dos desenlaces es válido y se reporta):

- *Réplica:* cosenos con KS y DPR del orden de lo reportado (≈ 0.97–0.98 y ≈ 0.94). Los valores absolutos de SED, PC, PR, PI y PE **no** serán iguales a los del artículo: el artículo mide piezas completas y aquí se miden los primeros acordes. El plan documenta cada diferencia.
- *Mejora:* C1 puede mejorar significativamente a C0 en SED y/o en alineación con KS sin perder DPR, o puede no mejorar. Con unas 120 evaluaciones del AG se explora poco, y se declara.

## Requisitos del Mac

- Apple Silicon (M1 o superior) y **macOS 14 o más reciente** (`torch` solo publica ruedas para esa combinación en Mac).
- 16 GB de RAM (la ejecución usa ≈ 10 GB con 3 procesos) y ≥ 15 GB de disco libre.
- Herramientas de línea de comandos de Apple (`xcode-select --install`: incluyen `git` y el compilador que necesita `dv`), Python ≥ 3.10 del sistema, [`uv`](https://docs.astral.sh/uv/) e internet.

## Estructura

```
README.md
pyproject.toml, uv.lock, env.sh   entorno reproducible (uv)
scripts/setup_vendor.sh           descarga BrainCog fijado a un commit (lo llama run_all.sh)
code/rep/                         el código de la réplica (config, patch, engine, evaluate, run_*, analyze, bench…)
data/paper_reference.json         valores reportados por el artículo (Fig. 5e, 6, 7, 8 y Tabla 2)
docs/GUIA_EJECUCION.md            guía para quien ejecuta
docs/PLAN_TECNICO.md              especificación completa: hallazgos, formulación, justificación de NSGA-II, diseño experimental
vendor/Brain-Cog/                 (no se sube; lo crea setup_vendor.sh)
results/, logs/                   (no se suben; los crea la ejecución)
```

El código del autor **no se modifica**: toda la parametrización va por *monkeypatch* en `code/rep/patch.py`, y una prueba de equivalencia (vendor puro vs. parche con valores por defecto) debe dar exactamente las mismas notas.

## Licencia

El código de este repositorio se publica bajo licencia **MIT** (ver [`LICENSE`](LICENSE)). El código del autor (BrainCog) no se redistribuye: se descarga aparte con `scripts/setup_vendor.sh` y conserva su licencia Apache-2.0.

## Referencias

- Liang, Zeng & Tang (2026). *A spiking neural network inspired by neuroscience and psychology for Western mode- and key-conditioned music learning and composition.* Sci Rep 16:12956. doi:10.1038/s41598-026-43529-1
- Zeng, Y. et al. (2023). BrainCog: a spiking neural network based, brain-inspired cognitive intelligence engine for brain-inspired AI and brain simulation. *Patterns* 4(8):100789. Código: https://github.com/BrainCog-X/Brain-Cog (Apache-2.0).
- Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans. Evol. Comput.* 6(2):182–197.
- Blank, J. & Deb, K. (2020). pymoo: Multi-objective optimization in Python. *IEEE Access* 8:89497–89509.
- Krumhansl, C. L. & Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys. *Psychological Review* 89(4):334–368.
