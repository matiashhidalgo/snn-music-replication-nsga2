> **Nota:** este documento es la especificación técnica escrita para Matías (autor del repositorio); "tu compañero" es quien ejecuta el código. Los números de sección (4.4, 5.3, 7…) se citan en comentarios del código.

# Plan de implementación · réplica de Liang, Zeng & Tang (2026) + NSGA-II de hiperparámetros

Proyecto: Bio-Inspired Computing, parte 2 · Paper: *A spiking neural network inspired by neuroscience and psychology for Western mode- and key-conditioned music learning and composition*, Sci Rep 16:12956, https://doi.org/10.1038/s41598-026-43529-1

**Para qué sirve este documento.** La sesión de construcción escribe el código **sin ejecutarlo**; después tu compañero lo corre en su Mac (M3/M4, 16 GB). Por eso aquí todo está especificado con exactitud y hay una **Fase A de verificación** (sección 7) que él corre primero. Lo que no pude verificar leyendo el código está marcado como **[GATE n]** y se comprueba en esa fase.

---

## 0. Las 5 garantías (dónde está cada una)

| Tu punto | Garantía | Dónde |
|---|---|---|
| 1. Justificación de NSGA-II | Argumento completo (problema, por qué multiobjetivo, por qué restricción, por qué NSGA-II y no otro, límites honestos) | §3.1 |
| 2. Outputs que explican todo | Mapa "pregunta de la rúbrica / sección del póster → tabla o figura" | §9.1 |
| 3. Todas las especificaciones | Matriz de trazabilidad contra cada viñeta de las instrucciones 1–7 y de la guía del póster | §11 |
| 4. Guardar todo, no re-correr | Cada evaluación se guarda **completa** (datos crudos); `analyze.py` solo lee archivos; `--check` verifica completitud; zip final | §5.7 |
| 5. Tu compañero sin sorpresas | Pre-vuelo automático, 2 fases con una sola orden cada una, reanudable, tabla de fallos y guía aparte (`02_guia_ejecucion.md`) | §8 |

---

## 0.1 Instrucciones para la sesión de construcción (leer primero)

Este documento es la especificación completa; no hay otra. Se trabaja dentro de `neural-music/`. **Solo se ejecuta la prueba rápida del paso 4**; nada largo (ni la Fase A completa, ni el AG, ni las 30 corridas).

**Reglas**
1. No editar `vendor/Brain-Cog/`. Todo cambio va en `code/rep/patch.py`.
2. Leer y reutilizar `code/c01_music_gen.py` y `code/metrics.py` antes de escribir `core.py`, `engine.py` y `metrics9.py`.
3. **No improvisar.** Este plan se escribió leyendo el vendor, sin ejecutarlo. Si un nombre, línea o atributo citado no existe o difiere (por ejemplo `pfc.modes.groups`, `syn.pre.areaName`, `computeFilterCurrent`, `doConnectToMode`, `temp >= 2`), leer el vendor, adaptar lo mínimo y anotarlo en la sección **Desviaciones** del informe (paso 4).
4. Rutas siempre relativas a `neural-music/` (el código correrá en otro Mac). Comentarios y mensajes en español, escuetos. Sin dependencias fuera de las de la sección 5.
5. Un solo lugar define el intérprete: `PY` en `run_all.sh` (el mismo mecanismo que ya usa el proyecto: `source env.sh` + `python`, o `uv run python`). La guía y `bench.py` usan ese mismo.
6. Carpeta de resultados configurable con `REP_RESULTS_DIR` (por defecto `results`). La prueba rápida usa `results_smoke`, para no ensuciar `results/` ni su caché.

**Paso 1 · Entorno (uv), sin instalar nada pesado**
```bash
cd neural-music && source env.sh
uv add --no-sync "pymoo>=0.6.1,<0.7" matplotlib
uv lock
```
`--no-sync` actualiza `pyproject.toml` y `uv.lock` **sin instalar** torch ni music21 (el Mac de tu compañero hará `uv sync`). Esos dos archivos van incluidos en lo que se manda. No borrar `.venv-vm` si existe.

**Paso 2 · Construir, en este orden:** `config.py` → `core.py` → `patch.py` → `data.py` → `engine.py` → `ksprofile.py` → `metrics9.py` → `evaluate.py` → `test_equiv.py` → `bench.py` → `run_replica.py` → `run_ga.py` → `run_final.py` → `analyze.py` → `run_all.sh` → `data/paper_reference.json` (valores de la tabla de T1, sección 9).

**Paso 3 · Comprobaciones sin SNN:** `python -m compileall -q code/rep`, `bash -n code/rep/run_all.sh`, y `--help` de cada script.

**Paso 4 · Pruebas sin SNN (única ejecución en la sesión de construcción)**

La sesión de construcción **no corre la SNN** (otro SO y arquitectura, poca RAM, sin entorno instalado). Lo que se prueba ahí es todo lo que no depende del vendor, con una evaluación simulada. La prueba con la SNN real (`--smoke`, S1–S10 de la sección 7) la corre tu compañero en la Fase A, y también tú si quieres adelantarla en tu Mac.

- `evaluate.py` obedece `REP_FAKE_EVAL=1`: devuelve un dict **con el mismo esquema** de 5.3, determinista a partir de (params, semilla), sin importar `vendor`, `torch` ni `music21` (`core.init_vendor()` se salta en ese modo). Con funciones suaves de los 6 parámetros, un conflicto suave entre `f1` y `f2` (para que el frente tenga varios puntos) y un `dpr` que a veces viola la restricción.
- Entorno ligero, solo para esto: un venv temporal fuera de `neural-music/` con `numpy scipy pandas matplotlib pymoo openpyxl` (≈ 500 MB, sin torch). Si ni eso se puede instalar, se omiten F4–F6 y se dice en el informe.
- `python code/rep/bench.py --selftest` (con `REP_RESULTS_DIR=results_selftest`) corre estas pruebas y escribe `results_selftest/informe_selftest.md`:

| Prueba | Qué comprueba |
|---|---|
| F1 | `metrics9`, `ksprofile`: `sed(x, x) = 0`; un caso con valor calculado a mano; `mm(KS_MAJOR)[0]=1`, `[2]≈0.30`, `[7]≈0.72`; `cos(v, v)=1`; pieza a mano de 4 voces en negras: EBR=0, GC=1, PpR=1 |
| F2 | caché: misma clave → no recalcula; cambiar `code_hash` la invalida; JSONL con última línea truncada se reanuda sin error |
| F3 | `Pool` *spawn* de 2 procesos con `maxtasksperchild=1` evalúa 2 individuos |
| F4 | `run_ga.py --pop 6 --gens 3`: fila 0 de la población inicial = `DEFAULTS`; existen `ga_config.json`, `history.jsonl` (4 generaciones) y `pareto.json` con ≥ 2 puntos; relanzar con otro `--hours` reutiliza `ga_config.json` y reescribe un `history.jsonl` idéntico |
| F5 | `run_replica.py` (30 semillas), `run_final.py`: elección de C1 por punto de rodilla; caso forzado sin factibles (`dpr` bajo) no se rompe y lo informa |
| F6 | `analyze.py --check` y `analyze.py` sobre esos datos simulados (30 semillas para C0 y C1, así Wilcoxon corre de verdad): T1–T5 y F1–F6 existen; el caso "todas las diferencias cero" da p = 1 sin excepción |
| F7 | `bash -n code/rep/run_all.sh` y `DRY_RUN=1 bash code/rep/run_all.sh B`: imprime las etapas en orden, con sus comandos y marcas `.done_<etapa>`, sin ejecutarlas |

El constructor agrega a mano la sección **Desviaciones** y **termina su respuesta pegando el informe completo** (`informe_selftest.md`), sin resumirlo. Su veredicto debe decir `SELFTEST PASS · SNN SIN PROBAR (se prueba en la Fase A)`.

**Paso 5 · Entrega:** lista de archivos creados, el informe del selftest pegado, y se detiene. Matías lleva el informe a esta conversación y, con él, se decide si se manda a tu compañero.

**Estructura fija de los informes** (`informe_selftest.md` y, en la Fase A, `informe_prueba.md`; `bench.py` los genera y el constructor solo añade Desviaciones al primero)
```
# INFORME DE PRUEBA RÁPIDA
## Entorno        arquitectura, SO, python, uv, torch, music21, pymoo, scipy, numpy, matplotlib, RAM, disco libre, code_hash
## Pre-vuelo      cada chequeo: PASS/FAIL + detalle
## Prueba rápida  S1–S10: PASS/FAIL, segundos y datos clave de cada una
## Archivos       ls de results_smoke con tamaños
## Desviaciones   (lo añade el constructor: qué cambió respecto al plan y por qué; "ninguna" si no hubo)
## Errores        últimas 40 líneas del log si algo falló
## Veredicto      SMOKE PASS|FAIL · LISTO PARA FASE A: sí|no
```

**Criterios que se revisan en `informe_prueba.md`** (la prueba con la SNN real de la Fase A; sirven para dar luz verde a la Fase B)

| Prueba | Debe cumplirse | Si falla |
|---|---|---|
| S2 equivalencia con el vendor | `gen_pitches` y `psc_major` idénticos | No mandar: el patch altera el modelo |
| S3 el parche engancha | cambiar `thr`, `izh_d`, `stdp_A` cambia el resultado | No mandar: los hiperparámetros no están conectados al modelo |
| S3 sin fuga de estado | evaluar `DEFAULTS` de nuevo da lo mismo que la primera vez | No mandar: el AG compararía individuos contaminados |
| S4 caché y S10 reanudación | segunda ejecución no recalcula (< 5 s) | Corregir antes de mandar |
| S5 Pool *spawn* | 2 evaluaciones en 2 procesos terminan | No mandar: el AG no correría en macOS |
| S6–S8 AG, final, analyze | generan todos los archivos de 5.7 y tablas T1–T5 y figuras F1–F6 sin excepción | Corregir antes de mandar |
| Valores (DPR, SED, cosenos) | **No se juzgan**: con 1 coral y L=2 no significan nada | — |

---

## 1. Decisiones

**Cerradas** (ya hablado con el profesor o decidido en sesión)
- La mejora es un algoritmo genético que ajusta hiperparámetros de la SNN (**aprobado por el profesor**). NSGA-II cuenta como AG (**aprobado**).
- Las 30 corridas se hacen sobre las **configuraciones finales** (C0 código por defecto vs. C1 afinada), con 30 semillas cada una, y el AG se ejecuta 1 vez (**aprobado**).
- No se edita `vendor/Brain-Cog/`. Todo cambio va por *monkeypatch* en `code/rep/patch.py`, y con parámetros por defecto debe dar **exactamente** lo mismo que el código original (test de equivalencia).
- Dataset principal: **Bach (music21)**. No necesita descarga ni parser nuevo.
- **Alcance obligatorio = C0 y C1** + el AG. Quedan **diferidos**: C0b (literal del paper), C2 (mínimo SED), búsqueda aleatoria de control, más ejecuciones del AG, SHTE. El código los deja preparados (flags) pero no se corren.
- La población inicial del AG incluye `DEFAULTS` como individuo 0 (más 11 aleatorios). Así el AG nunca queda peor que el punto de partida del autor y el frente siempre contiene una referencia comparable.

**Sin pendientes de aprobación.** Lo único que queda por decidir es el presupuesto de horas del AG, y lo fija el benchmark (sección 7).

---

## 2. Qué hace el código realmente (leído en `vendor/Brain-Cog/examples/Knowledge_Representation_and_Reasoning/musicMemory/`)

| # | Hecho | Dónde | Consecuencia para el diseño |
|---|---|---|---|
| H1 | El conocimiento de la tonalidad se inyecta con sinapsis clave→nota: diatónicas `20+U(20,30)` (excitatorias), cromáticas `-100` (inhibitorias). El comentario del autor dice que "aquí debería usarse el perfil KS" (no está hecho). | `Areas/cortex.py:107` `connectKeyAndNotesUsingKSModel` | El perfil KS **no** está inyectado; la alineación con KS debe surgir de otras sinapsis (H2). |
| H2 | PSC y PASW salen de las **sinapsis inversas** nota→neurona de modo (peso inicial 0, luego STDP). Se crean cuando hay coincidencia de disparos `temp >= 2`; **no hay chequeo de duplicados**, así que PSC cuenta coincidencias. | `Areas/pac.py:483` `doConnectToMode`, `pac.py:448` `doConnectToKey` | PSC/PASW ≈ histograma de alturas del conjunto de entrenamiento. Control barato obligatorio (sección 9). |
| H3 | El paper pide `o ≥ 5` para crear sinapsis; el código usa `temp >= 2`. | ídem | Discrepancia a documentar. Es hiperparámetro del AG. |
| H4 | Izhikevich: el código usa `d = 8` por defecto en todas las clases; el paper dice `d = 30`. `a, b, c, Vth` coinciden. | `Modal/izhikevichneuron.py` | Discrepancia a documentar. Es hiperparámetro del AG y define la config C0b. |
| H5 | STDP tipo 3 (clave→nota y nota→modo): `±5·exp(∓s/5)`, o sea A=5, τ=5 ms. Tipo 1 (entre grupos): amplitud 1, τ=5. El paper no da A± ni τ±. | `Modal/synapse.py` `computeWeight` | Son hiperparámetros del AG (solo tipo 3). |
| H6 | Entrenamiento: solo las **primeras 20 notas por voz** de cada coral (`part.flat.notes[:20]`); los grupos de neuronas están indexados por posición. Un coral tarda ≈12 s (40 corales ≈ 475 s en nuestra corrida anterior). | `api/music_engine_api.py` `rememberPartNotes` | La generación no puede pasar de `length = 20`. Reutilizar el entrenamiento de 40 corales del wrapper. |
| H7 | Generación: `generateMelodyWithKey(key_idx, firstNotes(4,1), None, L)`. Con `durations=None` todas las notas valen 1.0 (negra). Ganadora = neurona con más disparos entre las `preActive`; empate → índice menor. **No hay aleatoriedad** en la generación. | `Areas/cortex.py:468` | Todo el azar está en el entrenamiento (pesos iniciales, `random`). **[GATE 2]** medir cuánto varían los resultados entre semillas. |
| H8 | `generateMelodyWithKey` llama `pfc.doRememberingKey` → `update_learn` → modifica los pesos de las sinapsis inversas de las **neuronas de clave** (no las de modo, que son las que usamos). `setTestStates()` borra spikes y `strength`, así que el motor se puede reutilizar para varias piezas. | `Areas/pfc.py:255` | Generar varias piezas con un mismo motor entrenado. Por prudencia, extraer PSC/PASW **antes de generar**. **[GATE 1]** repetibilidad. |
| H9 | Neurona de modo `j` = semitonos sobre la tónica (las hojas `major`/`minor` de `inputs/keys.xlsx` son el orden cromático desde la tónica). Modo 0=mayor, 1=menor: `pfc.modes.groups[0|1]`. `keyIndexMap['C major']=0`, `'a minor'=1`; `idx%2`=modo, `idx//2`=fila. | `Areas/pfc.py:49`, `Modal/layer.py`, `inputs/keyIndex.csv` | PSC/PASW ya salen alineados a la tónica (como en Fig. 5). No hace falta transponer. |
| H10 | `code/c01_music_gen.py` ya resuelve: rutas relativas (`os.chdir(.../task)`), `sys.path`, y el *shim* de music21 (`Pitch(Pitch)`). `code/metrics.py` ya define DPR (mayor: escala mayor; menor: natural + 7.º elevado). | `code/` | Reusar, no reescribir. |

---

## 3. Formulación (para la sección "Problema y formulación" del póster)

- **Variables de decisión** `x ∈ R⁶` (tabla 4.3): amplitud STDP, constante de tiempo STDP, peso inicial medio, umbral de creación de sinapsis (entero), parámetro `d` de Izhikevich, amplitud de entrada de altura.
- **Objetivos (minimizar ambos):**
  `f1(x)` = SED entre las piezas generadas y una muestra de piezas reales (Ec. 16 del paper). Por defecto con las 9 métricas (`sed9`); si el benchmark detecta que el `ε` amplifica métricas casi constantes, con las 5 métricas de altura (`sed_pitch5`). Ver 4.4.
  `f2(x)` = 1 − coseno( PASW de modo mayor, perfil KS mayor ), ambos normalizados min-max (Fig. 5e).
- **Restricciones:** `DPR(x) ≥ 0.90` (promedio de las piezas generadas) y cotas `lo ≤ x ≤ hi`.
- **Algoritmo:** NSGA-II (Deb et al., 2002) con la implementación de `pymoo` (Blank & Deb, 2020), citada. Cruza SBX (η=15, p=0.9), mutación polinomial (η=20), restricción por dominancia (regla de Deb).
- **Evaluar `f(x)`** = entrenar una SNN completa con `x` y generar piezas: cada evaluación es una corrida de la SNN.

### 3.1 Por qué NSGA-II (para el póster y la defensa)

1. **El problema pide una metaheurística.** `f(x)` es una caja negra: cada evaluación entrena una SNN (minutos), no hay gradiente, el resultado depende de la semilla, el espacio es mixto (5 continuas + 1 entera) y para 3 de las 6 variables el paper ni siquiera da valor (H3–H5). Un algoritmo evolutivo de población no necesita derivadas, tolera la discontinuidad del umbral entero y paraleliza la población en 3 procesos.
2. **Hay dos criterios distintos y el paper los reporta por separado.** SED mide cercanía estadística a piezas reales (Fig. 8a); el coseno con KS mide consistencia con la percepción tonal humana (Fig. 5e). Sumarlos con pesos obligaría a inventar una ponderación sin base. Con dos objetivos se obtiene el **frente de Pareto**, que muestra si hay tensión entre ambos y permite ubicar el punto del paper en el mismo plano. Las instrucciones piden explícitamente una "formulación multiobjetivo" como mejora posible.
3. **La coherencia tonal es una restricción, no un objetivo.** DPR ≥ 0.90 refleja la idea central del modelo (Bach DPR 0.94 en Fig. 7b) y evita que el AG baje SED saliéndose de la escala. NSGA-II la maneja de forma nativa (regla de Deb: factible domina a no factible), sin factores de penalización arbitrarios.
4. **Por qué NSGA-II y no otro.** Es el AG multiobjetivo estándar (Deb et al., 2002), no necesita pesos ni vectores de referencia, maneja restricciones de forma nativa y existe una implementación validada y citable (`pymoo`) que hace el experimento reproducible. Alternativas descartadas: AG mono-objetivo con suma ponderada (pesos arbitrarios, una sola solución); MOEA/D (requiere vectores de pesos); PSO o DE (hay que añadir manejo de restricciones y multiobjetivo a mano); optimización bayesiana (no es bioinspirada).
5. **Cómo se evalúa el efecto.** C1 sale del frente (punto de rodilla) y se compara contra C0 (el código del autor) con 30 semillas que el AG nunca vio, con Wilcoxon pareado y corrección de Holm.

**Límites que se declaran, no se esconden.** (a) Con ≈120 evaluaciones y población 12, NSGA-II explora poco; se reporta tal cual. (b) Sin el control de búsqueda aleatoria (diferido) se puede afirmar "C1 supera a C0", pero no "NSGA-II supera a una búsqueda aleatoria del mismo costo". (c) Si SED y 1−cos no entran en conflicto, el frente se reduce a un punto: es un resultado válido y se informa (sección 10). (d) El fitness usa una semilla fija; el riesgo de sobreajuste se mide en las 30 semillas finales.

---

## 4. Diseño experimental

### 4.1 Perfiles de evaluación (`config.py`)

| Perfil | Uso | Corales de entrenamiento | Piezas de referencia | Piezas generadas | Largo L (acordes) |
|---|---|---|---|---|---|
| `fast` | dentro del AG | 5 | 10 | 6 | 6 |
| `full` | verificación y corridas finales | 40 (los primeros 40 `.mxl`, como el wrapper) | 10 | 10 | 8 |
| `smoke` | prueba de humo (~2 min) | 1 | 2 | 1 | 2 |

Valores iniciales; la Fase A mide tiempos y se pueden bajar (sección 8). Nota: el paper usa 10 piezas por modelo; el largo no lo reporta.

### 4.2 Semillas
- Cada corrida fija todas las fuentes: `seed_all(s)` = `random.seed`, `np.random.seed`, `torch.manual_seed`.
- **Fitness del AG: semilla fija `S_FIT = 1000 + ga_run_id`** (misma muestra de referencia y mismos pesos iniciales para todos los individuos). Así el fitness es determinista y la comparación entre individuos es justa. El riesgo de sobreajustar a esa semilla se cierra con la evaluación final en semillas **1–30**, que el AG nunca vio.
- Corridas finales: semillas 1–30.
- Lo que cambia con la semilla: pesos iniciales (H1) y qué 10 corales se muestrean como referencia.

### 4.3 Variables (`config.py: SPACE`)

| Clave | Qué es | Defecto (código) | Cota baja | Cota alta | Paper |
|---|---|---|---|---|---|
| `stdp_A` | amplitud STDP tipo 3 | 5.0 | 1.0 | 20.0 | no dice |
| `stdp_tau` | τ STDP tipo 3 (ms) | 5.0 | 2.0 | 20.0 | no dice |
| `w_init` | media del peso inicial clave→nota (`U(w−5, w+5)`) | 45.0 | 20.0 | 80.0 | no dice |
| `thr` | umbral de coincidencias para crear sinapsis inversa (entero; se redondea dentro de `evaluate`) | 2 | 1 | 6 | 5 |
| `izh_d` | `d` de Izhikevich | 8.0 | 2.0 | 30.0 | 30 |
| `alpha_p` | corriente de entrada de altura (`self.I`) | 30.0 | 15.0 | 60.0 | 30 |

Configuraciones a comparar:
- **C0 = "código por defecto"** (réplica): columna "Defecto".
- **C1 = afinada** (punto de rodilla del frente de Pareto, verificado en `full`). **Obligatoria.**
- C0b = "literal del paper" (C0 con `izh_d=30`, `thr=5`) y C2 = afinada de mínimo SED: **diferidas**; el código las soporta (`--config paper_literal`, `--pick min_sed`) pero no se corren.

### 4.4 Objetivos y restricción (por evaluación)
- `f1 = sed_f1`, `f2 = 1 − cos_major_pasw`, `g = 0.90 − dpr` (factible si `g ≤ 0`). `sed_f1` es `sed9` o `sed_pitch5` según `f1_metric` en `results/bench.json` (por defecto `sed9` si no existe).
- **Regla fijada de antemano para `f1_metric`** (la aplica `bench.py` en G2, antes del AG, sin intervención humana). Con `ε = 0.001`, una métrica cuya σ de referencia es ≈ 0 (PpR, EBR y GC en los primeros L acordes de corales de 4 voces en negras; quizá Pp) convierte cualquier diferencia mínima en un término enorme (0.04/0.001 = 40). Si se pierde un solo paso por silencio o un unísono, ese término domina `sed9` y el AG optimizaría artefactos y no la música. Regla: con `DEFAULTS` en `fast` y semillas 1–3, se calcula la contribución de cada métrica a `sed9²`; si las métricas con σ_ref < 0.01 suman **más del 50 %**, `f1_metric = "sed_pitch5"`; si no, `"sed9"`. Se guarda en `bench.json` y en `ga_config.json`, y **siempre se guardan y reportan ambos** (T1, T2 y T3 prueban `sed9` y `sed_pitch5`).
- Todo lo demás se registra sin entrar al AG: `sed_pitch5`, OA por métrica, `cos_*` (PSC/PASW, mayor/menor), `dpr`, las 9 métricas medias.
- Si una evaluación falla (excepción, sin neuronas activas): `f1=1e6, f2=1, dpr=0` y se registra el error. El AG sigue.

### 4.5 Métricas (`metrics9.py`). Definiciones elegidas, porque el paper no las fija

Pieza = 4 voces, cada una lista de `(altura MIDI, duración en negras)`; el onset es la suma acumulada de duraciones; rejilla de 0.25 negras; compás = 4 negras. Se redondea cada métrica a 2 decimales antes de promediar (como dice el paper).

| Métrica | Definición |
|---|---|
| PC | alturas MIDI distintas |
| PR | máx − mín (semitonos) |
| PI | media de \|Δaltura\| entre notas consecutivas dentro de cada voz |
| PE | entropía de Shannon (bits) del histograma de clases de altura (igual que `code/metrics.py`) |
| DPR | fracción de notas en la escala (regla de `code/metrics.py`) |
| Pp | polifonía media en los pasos con al menos una nota |
| PpR | fracción de pasos con 2 o más alturas simultáneas |
| EBR | fracción de negras sin ninguna nota sonando |
| GC | **1 − distancia de Hamming normalizada media** entre vectores de onset de compases vecinos (1 si hay menos de 2 compases). Se usa la consistencia (1−d) porque Fig. 7b da GC = 1.00 en Bach original y en el modelo; con la distancia pura saldría ≈ 0 |

**Referencia = los primeros L acordes** (primeras L notas de cada voz) de cada coral muestreado, con sus duraciones reales. Es lo justo: el modelo solo genera L acordes y solo aprendió 20 notas por voz. Cada pieza generada usa como semilla el primer acorde y la tonalidad de su coral de referencia.

SED (Ec. 16, **verificada en la p. 12 del PDF: lleva raíz cuadrada**): `sqrt( Σ_i (μgen_i − μref_i)² / (σref_i + 0.001)² )`, con `σ` con ddof=0 sobre las piezas de referencia.

Límites de comparabilidad con el paper (se documentan en T5): el paper mide piezas completas (PC=8.40 y PR=11.70 en Bach original) y nosotros solo los primeros L acordes, así que los valores absolutos de PC, PR, PI y PE no son comparables; sí lo son DPR, PpR, EBR, GC y los cosenos con KS. Fig. 7b da Pp = 4.19 en Bach original, imposible con 4 voces si se cuentan alturas distintas; el paper debe contar notas o solapes. Nosotros contamos alturas distintas y medimos referencia y generado con la **misma función**, así que el SED es autoconsistente.
OA: integral de `min(KDE_gen, KDE_ref)`; si alguna de las dos muestras tiene σ=0: 1 si las medias coinciden, 0 si no.

Como la generación siempre es polifónica de negras, Pp, PpR, EBR y GC dependen poco de los hiperparámetros (solo cambian con silencios o unísonos). Pero justamente por eso su σ de referencia puede ser ≈ 0 y `ε = 0.001` amplifica ×1000 cualquier diferencia: por eso la regla de `f1_metric` en 4.4. Se reportan igual.

---

## 5. Archivos a crear (`code/rep/`)

```
code/rep/
  config.py        PROFILES, SPACE (sección 4.3), DEFAULTS, rutas, S_FIT, listas de semillas
  core.py          init_vendor(): sys.path, chdir a .../musicMemory/task, shim music21, imports del vendor
  patch.py         apply(params, fast=False): monkeypatches (sección 6.1)
  data.py          índice de corales (cacheado), muestreo de referencia, eventos de una pieza
  engine.py        build_engine(params, n_train, seed), generate(eng, key, chord, L)
  ksprofile.py     KS_MAJOR, KS_MINOR, mm(), cos(), psc_pasw(eng, mode), histograma de control
  metrics9.py      9 métricas, sed(), oa()
  evaluate.py      evaluate(params, seed, profile) -> dict ; caché en disco ; CLI
  bench.py         --selftest (sin SNN, REP_FAKE_EVAL=1, F1–F7) · --preflight (30 s) · --smoke (S1–S10, ≈ 8–12 min) · completo: GATE 0-4 + tiempos + RAM → results/bench.json
  test_equiv.py    vendor puro vs patch con DEFAULTS: mismas notas y mismo PSC/PASW
  run_replica.py   30 semillas de una configuración (C0; C0b diferida) + control_hist.json
  run_ga.py        NSGA-II con pymoo, paralelo, reanudable
  run_random.py    búsqueda aleatoria con el mismo presupuesto (DIFERIDO: se escribe solo si se pide)
  run_final.py     verificar candidatos en full → elegir C1 → 30 semillas (C2 diferida)
  analyze.py       --check (completitud) · tablas, Wilcoxon y figuras (sección 9); solo lee results/
  run_all.sh       orquesta las fases A y B de la sección 8
results/           todo lo que se genera (JSONL, caché, figuras)
data/paper_reference.json   valores del paper (sección 9, T1), escrito a mano en la construcción
```

Agregar `pymoo` fijando versión: `uv add "pymoo>=0.6.1,<0.7"` y **subir `pyproject.toml` y `uv.lock`** (Matías lo hace antes de mandar nada; el Mac de tu compañero solo hace `uv sync`). `scipy`, `numpy`, `music21`, `torch`, `pandas`, `openpyxl` ya están. `analyze.py` además necesita `matplotlib` (`uv add matplotlib` si no está).

**Reglas generales de todos los scripts**
- `core.init_vendor()` se llama **al importar `engine.py`** (no en `main`): en macOS `multiprocessing` usa *spawn* y cada proceso hijo debe repetir el `chdir` y los imports.
- **Un hilo por proceso.** `core.init_vendor()` fija `torch.set_num_threads(1)` y `run_all.sh` exporta `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0`. Sin esto, 3 procesos con varios hilos cada uno se estorban y los tiempos del benchmark no valen. Los tiempos de G3 se miden ya con esta configuración.
- Pools con `maxtasksperchild=1` para liberar los ~3 GB de cada motor al terminar cada evaluación.
- Cada script escribe una línea JSON por resultado (append) y es **reanudable**: salta lo que ya existe. `evaluate` consulta una caché en `results/cache/<sha1(params redondeados + semilla + perfil + code_hash)>.json`, donde `code_hash` = sha1 de `config.py, patch.py, engine.py, data.py, ksprofile.py, metrics9.py, evaluate.py`. Si alguien edita el código, la caché vieja deja de valer sola. Relanzar el mismo comando reanuda.
- Cada script agrega a `results/manifest.json` (versiones de python/torch/music21/pymoo/numpy/scipy, `uname -m`, hostname, fecha, `code_hash`, argumentos).
- Todos aceptan `--procs N` (por defecto 3) y `--profile`. Si la suma de `--procs` de lo que corre a la vez pasa de 3, el script avisa y se detiene.

### 5.1 `data.py`
- `bach_index()`: recorre `m21.corpus.getComposer('bach')` filtrando `.mxl` (mismo orden que el wrapper), parsea **una vez** y guarda `code/rep/data/bach_index.json` con `{path, key, n_parts, n_notes, mode}`. Una pieza es válida si tiene 4 voces, `key in configs.keyIndexMap` y al menos 20 notas por voz.
- `ref_sample(seed, n, L)`: `random.Random(seed).sample(validos, n)`; devuelve para cada coral `key`, `chord` (la primera nota de cada voz, MIDI; si es acorde, la última nota, como hace `rememberPartNotes`) y `events` (primeras L notas por voz con duración real).
- `train_paths(n)`: los primeros `n` `.mxl` del orden del wrapper.

### 5.2 `engine.py`
- `build_engine(params, n_train, seed, vendor_only=False)`: `seed_all` → `patch.apply` (se salta si `vendor_only`, para `test_equiv.py`) → `EngineAPI(); cortexInit()` → para cada coral `rememberMusic` + `learnFourPartMusic` (igual que el wrapper). Devuelve el motor y el tiempo de entrenamiento.
- `generate(eng, key_str, chord, L)`: llama `eng.generateMelodyWithKey(configs.keyIndexMap[key_str], np.array([[p] for p in chord]), None, L)` y devuelve 4 listas de L alturas. **El primer acorde es la semilla**, así que se generan L−1 acordes nuevos. Un paso sin `"N"` es silencio (-1) y se omite en las métricas.

### 5.3 `evaluate.py`
```
evaluate(params, seed, profile) -> dict
  1. eng = build_engine(params_completo, cfg.n_train, seed)
  2. psc_pasw de modo 0 y 1               # ANTES de generar (H8)
  3. refs = ref_sample(seed, cfg.n_ref, cfg.L)
  4. gens = [generate(eng, r.key, r.chord, cfg.L) for r in refs[:cfg.n_gen]]
  5. M_gen (n_gen×9), M_ref (n_ref×9) → sed9, sed_pitch5, oa[9]
  6. cos_{major,minor}_{pasw,psc}, dpr (media sobre gens; tónica y modo con
     k = m21.key.Key(*key_str.split()) → k.tonic.pitchClass, k.mode)
  devuelve (TODO se guarda tal cual, ver 5.7): seed, profile, params, code_hash, versions,
            t_train, t_gen, sed9, sed_pitch5, oa[9], dpr, cos_{major,minor}_{pasw,psc},
            psc_{major,minor} (12), pasw_{major,minor} (12),
            M_gen (n_gen×9) y M_ref (n_ref×9) SIN promediar, ref_keys, gen_pitches (por pieza),
            metrics_gen_mean, metrics_ref_mean, error (None si todo bien)
```
`fitness(r) -> (r['sed9'], 1 - r['cos_major_pasw'], 0.90 - r['dpr'])`.
Los params se completan con `DEFAULTS` y `thr` se redondea a entero.

### 5.4 `run_ga.py`
- `--run-id 1 --pop 12 --gens 10 --profile fast --procs 3`.
- Problema **vectorizado** (`pymoo.core.problem.Problem(n_var=6, n_obj=2, n_ieq_constr=1, xl=LO, xu=HI)`), no `ElementwiseProblem` con runner: `_evaluate(self, X, out)` reparte la población con `POOL.map(eval_x, [decode(x) for x in X], chunksize=1)`, donde `POOL = multiprocessing.get_context("spawn").Pool(procs, maxtasksperchild=1)` es global del proceso principal y `eval_x` llama `evaluate(params, S_FIT, 'fast')`. (Un `ThreadPool` no sirve: el GIL anula el paralelismo y el *patch* es global.)
- Imports de pymoo 0.6.x: `pymoo.algorithms.moo.nsga2.NSGA2`, `pymoo.core.problem.Problem`, `pymoo.operators.crossover.sbx.SBX`, `pymoo.operators.mutation.pm.PM`, `pymoo.optimize.minimize`. Si alguno falla en la versión instalada, `bench.py --preflight` lo detecta antes de gastar horas.
- `--hours H` (por defecto 12) calcula `GENS` con la regla de la sección 7 usando `results/bench.json`; `--gens` explícito lo sobrescribe. **La primera vez** escribe `results/ga_run{ID}/ga_config.json` (pop, gens, semilla del AG, `S_FIT`, perfil, cotas, `DEFAULTS`, `code_hash`) y **al relanzar lo reutiliza e ignora `--hours`**, para que la reanudación sea idéntica.
- `NSGA2(pop_size=POP, sampling=X0, crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)`, semilla del AG = `run_id`. `X0` es un arreglo de numpy: fila 0 = `DEFAULTS`, filas 1..POP−1 = muestreo uniforme con `np.random.RandomState(run_id)` dentro de las cotas.
- *Callback* por generación: `results/ga_run{ID}/history.jsonl` con la población completa (X, F, G, dpr, factible), nº acumulado de evaluaciones, nº de factibles, tiempo. El archivo se **reescribe desde cero en cada lanzamiento** (las evaluaciones salen de la caché y pymoo es determinista con la misma semilla).
- Al final: `pareto.json` con los individuos factibles no dominados (X decodificado, F, dpr). Si no hay ninguno factible, se escribe `pareto.json` vacío con `"feasible": false` y `run_final.py` usa la mejor `g` y lo informa (no se rompe).

### 5.5 `run_final.py`
1. Lee `pareto.json`. Candidatos = punto de rodilla (mínima distancia al ideal (0,0) con objetivos min-max normalizados) + extremo de mínimo `f1` + 3 puntos repartidos por `f1` (los que existan; máximo 5).
2. **Verificación**: evalúa los candidatos **y `DEFAULTS`** en perfil `full` con `S_FIT` (cubre la diferencia entre `fast` y `full`, y deja C0 en el mismo plano que el frente) → `results/final/verify.jsonl`.
3. **C1** = el factible con menor distancia normalizada al ideal en `full` (**empate o ningún candidato factible**: se queda el de menor `f1`; si no mejora a `DEFAULTS` en `full`, igual se corre y se informa que el AG no mejoró). Guarda `results/final/C1.json`. C2 (`--pick min_sed`) diferida.
4. Corre 30 semillas (1–30) de C1 en `full` → `results/final/C1.jsonl`.

### 5.6 `run_replica.py`
`--config defaults|paper_literal --seeds 1-30 --profile full` → `results/replica/C0.jsonl` (o `C0b.jsonl`, diferida). Al empezar, si no existe, escribe `results/control_hist.json` (histograma de clases de altura relativas a la tónica de los 40 corales de entrenamiento, por modo, y su coseno con KS; sin SNN, ~1 min), para que `analyze.py` no necesite music21.

### 5.7 Persistencia: todo se guarda, nada se recalcula

```
results/
  preflight.json        verificación del Mac (sección 8)
  bench.json            G0–G4, tiempos, RAM, use_fast_patch, horas estimadas, pass_AB
  manifest.json         versiones, host, code_hash, argumentos (una entrada por etapa)
  control_hist.json     histograma vs KS (control sin SNN)
  cache/<sha1>.json     UNA evaluación completa (el dict de 5.3, íntegro)
  replica/C0.jsonl      30 líneas = 30 dicts de evaluate
  ga_run1/ga_config.json · history.jsonl · pareto.json
  final/verify.jsonl · C1.json · C1.jsonl (30 líneas)
  tables/  T1..T5 (.csv y .md) + versiones "poster" recortadas
  figs/    F1..F6 (.png 300 dpi y .pdf)
  results_bundle.zip    results/ + logs/ (lo que se manda de vuelta)
```

**Reglas**
1. **Datos crudos, no resúmenes.** Cada evaluación guarda `M_gen` y `M_ref` (9 métricas por pieza), `gen_pitches`, los vectores PSC/PASW de ambos modos, tiempos y errores. Con eso `analyze.py` puede recalcular SED, OA, cosenos, y variantes que a nadie se le ocurrieron hoy (SED sin GC, `f2` con el modo menor, otro `ε`), **sin volver a tocar la SNN**.
2. **`analyze.py` solo lee archivos.** No importa `vendor`, `torch` ni `engine`. Corre en el Mac de Matías sobre el zip y también sobre un `results/` generado por `--smoke` (así se prueba antes de gastar horas).
3. **Escritura segura.** JSONL con `flush` + `fsync` por línea; archivos sueltos con escritura a `.tmp` y `os.replace`. Al reanudar, una última línea truncada se descarta.
4. **`analyze.py --check`** imprime esperado vs. presente (C0: 30; C1: 30; AG: `GENS` generaciones; `verify` ≥ 2 líneas; `pareto.json` existe; `bench.json` con `pass_AB`; `control_hist.json`) y termina con código ≠ 0 si falta algo. `run_all.sh` lo corre al final, y solo si imprime `TODO COMPLETO` se arma el zip.
5. **Nada se borra.** Ni `cache/` ni `logs/`. Tamaño esperado del zip < 100 MB (verificar con `du -sh results` en G3).
6. **Qué obligaría a re-ejecutar la SNN.** Solo cambiar el modelo o el protocolo: cotas de `SPACE`, perfiles, definición de `f1`/`f2`/DPR, `n_ref`, `L` o la regla de métricas. Por eso esa lista se **congela antes de lanzar** (sección 8). Todo lo demás (pruebas estadísticas, figuras, criterios de coincidencia, tablas del póster) se rehace desde los datos guardados.

---

## 6. Código crítico

### 6.1 `patch.py`: parametrizar sin tocar el vendor
```python
import inspect, math, random, textwrap
from types import SimpleNamespace
from Modal import synapse as _syn, izhikevichneuron as _izh
from Areas import cortex as _cx, pac as _pac

DEFAULTS = dict(stdp_A=5.0, stdp_tau=5.0, w_init=45.0, thr=2, izh_d=8.0, alpha_p=30.0)
P = SimpleNamespace(**DEFAULTS)

def _rewrite(func, old, new, module):
    """Reescribe una línea literal del código original (así a los valores por defecto queda idéntico)."""
    src = textwrap.dedent(inspect.getsource(func))
    assert src.count(old) == 1, f"{func.__name__}: '{old}' aparece {src.count(old)} veces"
    ns = {}
    exec(src.replace(old, new), module.__dict__, ns)
    return ns[func.__name__]

_orig_cw = _syn.Synapse.computeWeight
def _computeWeight(self, t):                       # solo cambia el tipo 3 (A y tau parametrizados)
    if self.type != 3:
        return _orig_cw(self, t)
    for st in self.pre.spiketime:
        s = t - st - (self.delay - 1) * self.post.timeWindow
        self.weight += (P.stdp_A * math.exp(-s / P.stdp_tau) if s >= 0
                        else -P.stdp_A * math.exp(s / P.stdp_tau))

def _connectKey(self, keys, Notes):                # copia de connectKeyAndNotesUsingKSModel (cortex.py:107)
    noteNeurons = Notes.neurons
    for tone in keys.groups.values():
        tneurons = tone.neurons
        for nindex, noteneu in enumerate(noteNeurons[1:]):
            i = nindex % 12
            syn = _syn.Synapse(tneurons[i], noteneu)
            if tneurons[i].importance < 0:
                syn.excitability = 0; syn.weight = -100
            else:
                syn.excitability = 1
                syn.weight = (P.w_init - 25) + random.uniform(20, 30)   # defecto: 20 + U(20,30), idéntico
            syn.type = 3
            noteneu.synapses.append(syn); noteneu.pre_neurons.append(tneurons[i])

_installed = False
def apply(params=None, fast=False):
    global _installed
    for k, v in {**DEFAULTS, **(params or {})}.items():
        setattr(P, k, int(round(v)) if k == "thr" else float(v))
    _izh.P = P; _pac.P = P
    if not _installed:
        _syn.Synapse.computeWeight = _computeWeight
        _cx.Cortex.connectKeyAndNotesUsingKSModel = _connectKey
        M = _pac.Music_Sequence_Mem
        M.doConnectToKey  = _rewrite(M.doConnectToKey,  "temp >= 2", "temp >= P.thr", _pac)
        M.doConnectToMode = _rewrite(M.doConnectToMode, "temp >= 2", "temp >= P.thr", _pac)
        N = _izh.NoteIzhikevichNeuron
        N.computeFilterCurrent = _rewrite(N.computeFilterCurrent, "self.I = 30", "self.I = P.alpha_p", _izh)
        _installed = True
    for cls in vars(_izh).values():                # d de Izhikevich: se reasigna en CADA apply, antes de crear neuronas
        if isinstance(cls, type) and issubclass(cls, _izh.IzhikevichNeuron) and "__init__" in cls.__dict__:
            d = cls.__init__.__defaults__
            if d and len(d) >= 5:
                cls.__init__.__defaults__ = d[:3] + (P.izh_d,) + d[4:]
    if fast:                                       # opcional (GATE 3): equivalente exacto, spiketime ascendente en generación
        def _stf2(self, t):
            sp = self.pre.spiketime
            if self.type == 1:
                if sp and sp[0] + self.delay <= t: self.strength = 1
            elif self.type >= 2:
                if sp and sp[0] <= t: self.strength = 1
        _syn.Synapse.computeShortTermFacilitation2 = _stf2
```
`_rewrite` ejecuta la fuente del método original con una sola línea cambiada. Si el vendor cambia o la línea no existe, falla con un `assert` claro.

### 6.2 `ksprofile.py`: PSC/PASW (H2, H9)
```python
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])   # Krumhansl 1990
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def mm(v):
    v = np.asarray(v, float); r = v.max() - v.min()
    return (v - v.min()) / r if r > 0 else np.zeros_like(v)

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

def psc_pasw(eng, mode):                            # mode 0 = mayor, 1 = menor
    grp = eng.cortex.pfc.modes.groups[mode]
    psc, wsum = np.zeros(12), np.zeros(12)
    for j, neu in enumerate(grp.neurons):           # j = semitonos sobre la tónica
        for syn in neu.synapses:
            if syn.pre.areaName == "NMSM":          # neurona de altura de la SMS (sinapsis inversa)
                psc[j] += 1; wsum[j] += syn.weight
    pasw = np.divide(wsum, psc, out=np.zeros(12), where=psc > 0)
    return psc, pasw
```
`cos_major_pasw = cos(mm(pasw_major), mm(KS_MAJOR))`; igual para PSC y para el menor. Verificado contra `code/charts/ps_major.csv`: `mm(KS_MAJOR)` reproduce la columna `ks` (C=1.0, D=0.30, G=0.72).

**Control sin SNN** (lo calcula `run_replica.py` y lo guarda en `control_hist.json`; `analyze.py` lo lee): `cos(mm(histograma de clases de altura relativas a la tónica, de los corales de entrenamiento, por modo), mm(KS))`. Si da ≈ 0.98, la "alineación emergente" es solo conteo de alturas. Se reporta como hallazgo.

### 6.3 `metrics9.py`: SED y OA
```python
def sed(gen, ref, eps=1e-3):                        # gen (n_gen×9), ref (n_ref×9), ya redondeados a 2 decimales
    mu_g, mu_r, sd_r = gen.mean(0), ref.mean(0), ref.std(0)
    return float(np.sqrt((((mu_g - mu_r) / (sd_r + eps)) ** 2).sum()))

def oa(g, r, n=2000):
    if g.std() == 0 or r.std() == 0:
        return 1.0 if abs(g.mean() - r.mean()) < 1e-9 else 0.0
    x = np.linspace(min(g.min(), r.min()) - 1, max(g.max(), r.max()) + 1, n)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.minimum(gaussian_kde(g)(x), gaussian_kde(r)(x)), x))
```

---

## 7. Fase A · `bench.py` (la corre tu compañero primero)

Tres pasos, en este orden:
1. `bench.py --preflight` (~30 s). Comprueba: macOS en `arm64`; RAM total ≥ 15 GB (`sysctl hw.memsize`); disco libre ≥ 15 GB; `uv`, `caffeinate`, `zip` presentes; el Mac está enchufado (`pmset -g batt`, solo advertencia); importan `torch, music21, scipy, numpy, pandas, openpyxl, matplotlib, pymoo` (imprime versiones) y los imports de pymoo de 5.4; existen `vendor/.../musicMemory/inputs/keys.xlsx` y `keyIndex.csv`; el corpus music21 de Bach da ≥ 40 `.mxl` válidos. Escribe `results/preflight.json`.
2. `bench.py --smoke` (≈ 8–12 min, sin medir): todo el *pipeline* con perfil `smoke` (1 coral, L=2), escribiendo en `results_smoke/` (nunca en `results/`). Pruebas, cada una con PASS/FAIL, segundos y datos clave en el informe (0.1):
   - **S1** `evaluate(DEFAULTS, seed 1, smoke)` termina y devuelve todos los campos de 5.3, sin error.
   - **S2** equivalencia mini: vendor puro (`vendor_only=True`) vs `patch.apply(DEFAULTS)`, misma semilla: mismas `gen_pitches` y mismo `psc_major`.
   - **S3** el parche engancha y no hay fuga de estado: `evaluate` con `thr=1`, `izh_d=30`, `stdp_A=10` (uno por uno) cambia el resultado respecto a `DEFAULTS`, y volver a evaluar `DEFAULTS` en el mismo proceso da exactamente lo de S1.
   - **S4** caché: repetir S1 sale de la caché (< 2 s) con el mismo dict.
   - **S5** `Pool` *spawn* con 2 procesos y `maxtasksperchild=1` evalúa 2 individuos distintos.
   - **S6** `run_ga.py --pop 4 --gens 1 --profile smoke --procs 2`: crea `ga_config.json`, `history.jsonl` (2 generaciones) y `pareto.json` (puede salir vacío con `feasible: false`).
   - **S7** `run_replica.py --seeds 1-2 --profile smoke` y `run_final.py --verify 2 --seeds 1-2 --profile smoke`: crean `C0.jsonl`, `verify.jsonl`, `C1.json`, `C1.jsonl` (2 líneas cada JSONL).
   - **S8** `analyze.py --check` y `analyze.py` sobre `results_smoke/`: T1–T5 y F1–F6 sin excepción (con 2 semillas Wilcoxon no significa nada, pero no debe fallar).
   - **S9** `control_hist.json` se generó y `data/paper_reference.json` trae todas las claves de la sección 9.
   - **S10** reanudación: relanzar S7 no recalcula (dice cuántas existían y termina en < 5 s).

**Probablemente falle algo la primera vez** (el código nunca corrió contra el vendor). Cada intento de la Fase A cuesta ≈ 10 min, así que el costo real es la ida y vuelta, no el cómputo. Para acortarla, cada fallo viene con su causa más probable y quién actúa:

| Falla | Causa más probable | Quién y qué |
|---|---|---|
| Preflight: falta un import o `uv sync` falla | Entorno del Mac | Tu compañero: repetir `uv sync`; si persiste, mandar `logs/` |
| Preflight: RAM o disco insuficientes | El Mac está justo | Tu compañero: cerrar apps y liberar disco; repetir |
| S1: excepción en `build_engine` o `generate` | Nombre o firma del vendor distinto a lo citado en la sección 2 | Matías/constructor: leer el traceback y el vendor, corregir `engine.py` |
| S1: `psc_major` todo ceros | `syn.pre.areaName == "NMSM"` no coincide, o `thr` demasiado alto con 1 coral | Matías/constructor: imprimir los `areaName` de las sinapsis de `pfc.modes.groups[0]` y ajustar `ksprofile.py` |
| S2: `gen_pitches` o `psc_major` distintos al vendor | Orden de llamadas a `random` distinto en `_connectKey`, o `izh_d` mal reasignado | Matías/constructor: no continuar; revisar `patch.py` línea por línea contra `cortex.py:107` |
| S3: cambiar un parámetro no cambia nada | La reescritura no quedó instalada, o `P` no es el mismo objeto en `_pac`/`_izh` | Matías/constructor: revisar `apply()` y los `assert` de `_rewrite` |
| S3: `DEFAULTS` da algo distinto la segunda vez | Estado global que sobrevive entre evaluaciones (contadores de clase del vendor, `P`) | Matías/constructor: reiniciar el estado en `build_engine`; el Pool con `maxtasksperchild=1` ya lo aísla en la corrida real |
| S5/S6: se cuelga o `ModuleNotFoundError` en el hijo | `init_vendor()` no corre al importar `engine.py`, o falta el `if __name__ == "__main__"` | Matías/constructor: corregir `core.py` o el script |
| S6–S8: error de `pymoo` | Versión o ruta de import distinta | Matías/constructor: ajustar el import; el `uv.lock` fija la versión |
| Cualquier otra | — | Tu compañero **no edita código**: manda `results_smoke/informe_prueba.md` y `logs/`; Matías corrige y le manda los archivos nuevos |
3. `bench.py --procs 3` (~20–30 min): G0–G4. Escribe `results/bench.json` con PASS/FAIL por gate, `use_fast_patch`, `t_eval_fast`, `t_eval_full`, RAM pico, `horas_estimadas` por bloque y `pass_AB` (= G0 y G4 pasan; G1–G3 solo informan y fijan parámetros).

| Gate | Qué comprueba | Si falla |
|---|---|---|
| G0 | Entorno: `uv sync` ok; `evaluate(DEFAULTS, 1, 'fast')` termina; `dpr ≥ 0.9`; `psc_major.sum() > 0`; el entrenamiento de 40 corales contiene ≥ 8 corales mayores y ≥ 8 menores | Corregir entorno o elegir otros corales de entrenamiento |
| G1 | Repetibilidad: dos `generate` seguidos con el mismo motor, tonalidad y acorde dan notas idénticas. Informativo: si `psc_major`/`pasw_major` cambian entre antes y después de generar (no deberían, H8) | Si las notas no son idénticas, generar cada pieza con un motor nuevo (costo mucho mayor, reportarlo) |
| G2 | Varianza entre semillas: `evaluate(DEFAULTS, s, 'fast')` para s=1,2,3. Reporta cuánto varían `sed9`, `cos_major_pasw`, `gen_pitches`, la contribución de cada una de las 9 métricas a `sed9²` y **fija `f1_metric`** con la regla de 4.4 | Si casi no varían, las 30 corridas darán muchos empates; se reporta y la variación viene de la muestra de referencia |
| G3 | Tiempos y RAM: `t_train(5)`, `t_train(40)`, `t_gen` por pieza (L=8), RAM pico por proceso, y la aceleración de `fast=True` con salidas idénticas | Adoptar `fast=True` solo si acelera ≥ 1.3× y `gen_pitches` es idéntico |
| G4 | `test_equiv.py`: vendor puro (`--vendor-only`, sin `patch.apply`) vs `patch.apply(DEFAULTS)`, 3 semillas: mismas `gen_pitches` y mismo `psc_major`. **Ya resuelto en esta sesión:** la Ec. 16 sí lleva raíz y los valores de las Fig. 5e, 6, 7 y 8 y la Tabla 2 están en la sección 9 (T1), para copiarlos a `data/paper_reference.json`; `bench.py` solo comprueba que el archivo existe y trae todas las claves | El patch no es equivalente: no continuar hasta corregirlo |

**Presupuesto del AG (se fija con G3):** `N_eval = floor(H_ga · 3600 · P / t_eval_fast)`, con `H_ga` las horas de reloj que quieran dedicar y `P` los procesos. Luego `POP = 12` y `GENS = max(5, floor(N_eval / POP) − 1)`.

---

## 8. Cómputo, orden de ejecución y qué recortar

**Estimación inicial, sin medir** (parte de nuestras corridas anteriores: ≈12 s por coral y ≈26–32 s por paso de generación; el bench las reemplaza)

| Bloque | Cuenta | Reloj con 3 procesos |
|---|---|---|
| Evaluación `fast` | ≈1 min entrenar + 6 piezas × 6 pasos × 28 s ≈ 18 min | |
| AG (POP 12 × 10 generaciones ≈ 120 evaluaciones) | 120 × 18 min | ≈ 12 h |
| Verificación de 5 candidatos en `full` | 5 × 45 min | ≈ 1.3 h |
| C0: 30 semillas en `full` (8 min + 10 piezas × 8 pasos × 28 s ≈ 45 min c/u) | 30 × 45 min | ≈ 7.5 h |
| C1: 30 semillas | igual | ≈ 7.5 h |
| **Obligatorio** | | **≈ 28 h** |
| Extras **diferidos**: C0b, C2 (7.5 h c/u), control aleatorio (12 h), más ejecuciones del AG (12 h c/u) | | no se corren por ahora |

RAM: 3 procesos × ≈3.3 GB ≈ 10 GB (medido con 40 corales; **verificar** con G3).

**Cómo recortar si se queda corto:** bajar `L` (8→6), `n_gen` (10→6), `POP`/`GENS` (`GA_HOURS`). Mantener siempre las 30 semillas de C0 y C1. Esto se decide con `bench.json` **antes** de la Fase B y se congela en `ga_config.json`.

### 8.1 Congelar antes de lanzar (checklist para Matías, al terminar de construir)

Cambiar cualquiera de estos después obliga a re-ejecutar horas de SNN (sección 5.7, regla 6). Confirmarlos **una vez**, antes de mandar el código:
- [ ] Cotas y valores por defecto de `SPACE` (4.3) y qué parámetros entran al AG.
- [ ] Perfiles `fast`/`full` (4.1): `n_train`, `n_ref`, `n_gen`, `L`.
- [ ] `f1`, `f2`, restricción `DPR ≥ 0.90` (4.4) y definiciones de las 9 métricas (4.5).
- [ ] `S_FIT = 1000 + run_id` y semillas finales 1–30.
- [ ] `data/paper_reference.json` completo (sección 9, T1).
- [ ] `uv add "pymoo>=0.6.1,<0.7" matplotlib`, `uv lock`, y `pyproject.toml` + `uv.lock` incluidos en lo que se manda.
- [ ] **Repositorio de GitHub (sustituye al empaque manual).** La carpeta `neural-music/` es el repositorio. Su `.gitignore` es una lista blanca: solo suben `README.md`, `pyproject.toml`, `uv.lock`, `env.sh`, `scripts/`, `code/rep/`, `data/paper_reference.json` y `docs/`. Nada de la parte 1 (video, audios, imágenes), ni entornos, cachés, resultados o el vendor. El código del autor lo baja `scripts/setup_vendor.sh` (commit fijado `f9b879f`) cuando se corre la Fase A. Matías crea el repo (recomendado: **privado**, con tu compañero como colaborador), antes del primer `git add` revisa la lista con `git status`, y le pasa el enlace.
- [x] ~~Prueba en el Mac de Matías~~ **No es posible**: es un Mac Intel (`macosx_x86_64`) y `torch` 2.14.1 no publica ruedas para esa plataforma (solo `macosx_14_0_arm64`, Linux y Windows). **La primera ejecución real con la SNN será la Fase A de tu compañero** (Apple Silicon, macOS ≥ 14), por eso importan la tabla de fallos de la sección 7 y el informe `informe_prueba.md`.

### 8.2 Dos fases, una orden cada una (`code/rep/run_all.sh`)

Se mandan a tu compañero tres cosas: la carpeta `neural-music/`, este flujo y la guía `replication/02_guia_ejecucion.md` (documento aparte, escrito para él). Las etapas corren **una tras otra**; la suma de `--procs` de lo que corra a la vez no pasa de 3 (≈10 GB de RAM).

```bash
# FASE A (≈ 30–40 min)
bash code/rep/run_all.sh A
#   source env.sh && uv sync · exporta las variables de un hilo · bench.py --preflight · --smoke · --procs 3
#   termina con "FASE A OK" (pass_AB = true) o con "FASE A FALLÓ: <gate>" y código ≠ 0
#   → mandar results/bench.json y logs/ y esperar luz verde

# FASE B (≈ 28 h, sin atención)
nohup caffeinate -dimsu bash code/rep/run_all.sh B > logs/run_all.log 2>&1 &
```

Fase B exige `pass_AB = true` y corre estas etapas; cada una escribe `results/.done_<etapa>` al terminar bien y se **salta** si ya existe, así que relanzar la misma orden reanuda:

| Etapa | Comando (todos con `--procs ${PROCS:-3}` y `2>&1 \| tee logs/<etapa>.log`) |
|---|---|
| `c0` | `run_replica.py --config defaults --seeds 1-30` |
| `ga1` | `run_ga.py --run-id 1 --pop 12 --hours ${GA_HOURS:-12} --profile fast` |
| `final` | `run_final.py --ga-run 1 --verify 5 --seeds 1-30` |
| `check` | `analyze.py --check && analyze.py` |
| `zip` | `zip -r results_bundle.zip results logs` (solo si `check` imprimió `TODO COMPLETO`) |

`run_all.sh` escribe fecha y hora de inicio y fin de cada etapa en `logs/run_all.log`. Si una etapa falla, se detiene, imprime cuál y qué hacer, y no sigue con las siguientes. Si el Mac solo aguanta 2 procesos (por ejemplo, `Killed` o falta de memoria), `PROCS=2 bash code/rep/run_all.sh B` (con `nohup caffeinate` igual).

### 8.2b Hallazgos de la sesión de construcción (informe del selftest)

| Hallazgo | Qué hacer |
|---|---|
| `izh_d` no afecta a las neuronas de tonalidad, solo a las clases que usan ese parámetro | Documentar en T5 qué clases cambia. En S3, si cambiar `izh_d` no altera **nada** del resultado, es una dimensión inerte: se retira del AG (5 variables) y se anota |
| `uv.lock` regenerado: 134 paquetes, incluye `dv` (sdist), `timm`, `tonic`, `pygame`, `thop` | `dv` se compila en `uv sync`: hace falta Xcode Command Line Tools (`xcode-select --install`). El preflight comprueba `xcode-select -p` y la guía lo pide |
| `env.sh` fija `UV_PYTHON_PREFERENCE=only-system` | Hace falta un Python del sistema ≥ 3.10 (el `python3` de macOS suele ser 3.9). El preflight imprime `python3 --version`; si es < 3.10 e imprime la instrucción: `brew install python@3.12` (o `export UV_PYTHON_PREFERENCE=managed` para que `uv` descargue uno) |
| "Bug de la rama T del vendor" sin tocar | El constructor debe describirlo en una línea en T5 y confirmar si afecta a nuestro flujo (generación con `durations=None`) |
| `.uv-cache` de ≈ 12 MB en `neural-music/` | No hace falta borrarlo; no pedir permiso de borrado. Se excluye del zip que se manda |
| `bench.json` simulado dentro de `results_selftest/` | Correcto; `results_selftest/` y `results_smoke/` no se mandan como resultados finales |

### 8.3 Qué hace que "no haya sorpresas"

| Riesgo en el Mac de tu compañero | Cómo se cubre |
|---|---|
| Dependencias faltantes o versiones distintas | `uv sync` desde `uv.lock`; `--preflight` importa todo e imprime versiones |
| El Mac se suspende y se pierde el trabajo | `caffeinate -dimsu`; todo reanudable; la guía pide enchufarlo y no cerrar la tapa |
| Cerrar la Terminal mata el proceso | `nohup … &` y `tail -f logs/run_all.log` |
| Se acaba la RAM (`Killed`) | `maxtasksperchild=1`, un hilo por proceso, `PROCS=2` de reserva |
| Disco lleno | `--preflight` exige ≥ 15 GB libres |
| Tarda mucho más de lo previsto | La Fase A mide los tiempos reales, imprime `horas_estimadas` y fija `GENS`; se decide antes de gastar 28 h |
| El patch cambia el modelo sin que nadie lo note | G4 (equivalencia con el vendor) bloquea la Fase B |
| Resultados incompletos o corruptos | Escritura atómica, `--check`, zip solo si todo está |
| Reanudar con código distinto | `code_hash` en la clave de caché |

---

## 9. Qué sale de `analyze.py` (alimenta análisis y póster)

**Tablas** (CSV + markdown)
- **T1 Réplica vs paper (Bach).** Para cada valor reportado: media, mediana, σ y [mín, máx] de nuestras 30 corridas de C0, el valor del paper y el veredicto. **Valores del paper, ya leídos** (copiar tal cual a `data/paper_reference.json`; los de figuras se leyeron de la imagen y valen ±0.01):

| Qué | Dónde | Valor del paper (Bach) |
|---|---|---|
| Coseno PASW–KS, modo mayor / menor | Fig. 5e | 0.98 / 0.94 |
| Coseno PSC–KS, modo mayor / menor | Fig. 5e | 0.97 / 0.94 |
| PSC(G), PASW(G) en modo mayor (puntaje normalizado de la neurona G) | texto p. 11 | 0.98, 0.90 (la Fig. 5c rotula 1.0 y 0.95: inconsistencia del paper, se documenta) |
| Coseno por tonalidad, C mayor (PSC, PASW) | Fig. 6 | 0.93, 0.97 |
| Medias de las 9 métricas, datos originales (PC, PR, PI, DPR, PE, Pp, PpR, EBR, GC) | Fig. 7b | 8.40, 11.70, 1.92, 0.93, 4.29, 4.19, 1.00, 0.00, 1.00 |
| Medias de las 9 métricas, "Our Model" | Fig. 7b | 4.50, 9.00, 2.20, 0.94, 3.63, 3.95, 0.96, 0.00, 1.00 |
| OA de "Our Model" (mismo orden) | Tabla 2 | .71, .82, .71, .83, .90, .48, .00, 1.00, .20 |
| SED Bach: Our Model, Diffusion, LSTM, CNN, GAN, RNN, Transformer | Fig. 8a | 9.24, 40.66, 204.77, 194.48, 177.66, 171.46, 143.22 |

Referencia SHTE (solo contexto, no se replica): cosenos Fig. 5e mayor 0.98/0.98 y menor 0.92/0.93 (PASW/PSC), SED de Our Model 30.11.

**Qué es "replicar" aquí.** Comparable en valor: cosenos con KS, DPR, PpR, EBR, GC. Comparable solo en orden de magnitud y tendencia: SED (referencia y largo distintos, 4.5) y PC, PR, PI, PE (piezas completas vs. L acordes). **Criterio propuesto:** coincide si el valor del paper cae dentro de [mín, máx] de las 30 corridas y la diferencia de medianas es ≤ 0.05 (DPR, cosenos, GC, PpR, EBR); para SED, mismo orden de magnitud. Se acompaña de Wilcoxon de una muestra contra el valor del paper. Como `analyze.py` solo lee datos guardados, el criterio se puede cambiar sin re-correr nada.
- **T2 Estadística por configuración** (C0, C1; C0b y C2 si algún día se corren): media, mediana, σ, mín, máx de `sed9`, `1−cos_major_pasw`, `dpr`, OA medio, y cada métrica.
- **T3 Pruebas.** Tres comparaciones, que son las que pide la instrucción 5 (réplica, artículo y mejora):
  - (a) **Réplica vs artículo:** C0 contra cada valor del paper (Wilcoxon de una muestra).
  - (b) **Mejora vs réplica:** C0 vs C1 pareado por semilla (la semilla fija muestra de referencia y pesos iniciales) sobre `f1`, `f2` y `dpr`, con corrección de Holm; tamaño de efecto (diferencia de medianas y *rank-biserial*).
  - (c) **Mejora vs artículo:** C1 contra los valores del paper (Wilcoxon de una muestra).
  Si todas las diferencias son cero, p = 1 (manejar el error de `scipy`). Si la varianza entre semillas es 0 (G2), se reporta como tal y no se finge significancia.
- **T4 Parámetros.** Valores de C1 contra defecto del código y valor del paper.
- **T5 Replicabilidad y dificultades.** Una fila por discrepancia o limitación (H1–H5, solo 20 notas por voz, 40 de ≈408 corales, duraciones siempre negras, baselines no replicables por parámetros en el Suplemento S2, métricas ambiguas, GC como 1−d, Pp, anomalía Fig. 5b/5d, inconsistencia Fig. 5c vs texto), con el impacto y el número medido cuando aplica (por ejemplo, el coseno del histograma de control).
- **Versiones "póster"** de T1 y T3 (`tables/*_poster.md`): ≤ 4 columnas × ≤ 7 filas, como pide la guía del póster.

**Figuras**
- F1 convergencia por generación: mejor `f1`, mejor `f2`, nº de factibles e hipervolumen (de `history.jsonl`; punto de referencia del HV = (1.1·máx `f1` factible visto, 1.0)).
- F2 frente de Pareto en `full` con los puntos C0, C1 y el punto del paper (9.24, 1 − 0.98), con la advertencia de que no es del todo comparable (4.5).
- F3 PSC/PASW normalizados vs KS mayor, promedio de 30 corridas, C0 y C1 (estilo Fig. 5).
- F4 diagramas de caja de las 30 corridas para C0/C1 (`f1`, `f2`, `dpr`).
- F5 control del histograma (arriba): coseno histograma–KS vs coseno SNN–KS.
- F6 las 9 métricas medias: datos originales (paper), Our Model (paper), C0 y C1.

### 9.1 Mapa: qué output explica qué

| Lo que hay que poder explicar | Output | Sección del póster |
|---|---|---|
| Problema, variables, objetivos, restricción | §3, T4 (espacio de búsqueda) | Problema y formulación |
| Por qué NSGA-II | §3.1 | Algoritmo y metodología |
| Hiperparámetros del AG y protocolo | `ga_config.json`, `manifest.json`, §4 | Algoritmo y metodología |
| Réplica contra los resultados originales | T1, F3, F6 | Resultados de la réplica |
| Comportamiento del AG y su efectividad (instr. 6a) | F1, F2 (frente) | Mejora propuesta |
| Mejora frente a réplica y artículo, con estadística (instr. 5) | T2, T3 (a, b, c), F4 | Mejora propuesta y sus resultados |
| ¿Es la "alineación con KS" un resultado del modelo o solo conteo de alturas? | F5, `control_hist.json` | Conclusiones |
| Qué tan replicable fue el artículo (instr. 6b) | T1 (veredictos) y T5 | Conclusiones |
| Dificultades encontradas (instr. 6c) | T5, `bench.json` (tiempos y RAM), §2 | Conclusiones |

**Para el póster** (guía: ≤ 800 palabras, ≈ 3 gráficos, tablas de ≤ 4 × 7, 4–5 referencias, conclusiones arriba o a la derecha): gráficos = F3, F2 y F4; tablas = T1 y T3 en versión "póster". F1, F5 y F6 quedan de respaldo o de apoyo en la defensa.

**Para "qué tan replicable fue" y "dificultades"** (T5): H3, H4, H5 (parámetros que el paper no da o que difieren del código), H2/H8 (PSC/PASW ≈ conteo; se miden antes de generar), solo 20 notas por voz y 40 de ≈408 corales, duraciones siempre negras, baselines no replicables (parámetros en el Suplemento S2), métricas ambiguas (sección 4.5), y la anomalía de las Fig. 5b y 5d (idénticas punto por punto aunque sus cosenos en Fig. 5e difieren; por eso `f2` usa solo modo mayor).

---

## 10. Riesgos y extensiones

| Riesgo | Mitigación |
|---|---|
| Las semillas casi no cambian los resultados (H7, G2) | Se reporta; la variación viene de la muestra de referencia. Wilcoxon con empates se maneja. |
| Los dos objetivos no entran en conflicto y el frente se reduce a un punto | Se verá en las primeras generaciones; se reporta como resultado y se sigue con `f1` + restricción. |
| El SED no será numéricamente comparable al 9.24 (métricas ambiguas) | Se compara orden de magnitud, ranking entre configuraciones y, de los 9 factores, los de altura. Se documenta. |
| Proxy `fast` ≠ `full` | Paso de verificación en `full` antes de elegir C1. |
| `DPR` ya ≈ 1.0: poco margen de mejora | Resultado válido: "el AG mejora SED/KS sin perder DPR" o "no mejora significativamente". Ambos se reportan con Wilcoxon. |
| Instalación de pymoo/torch en el Mac de tu compañero | `uv sync` desde el `uv.lock`; G0 lo detecta antes de gastar horas. |
| El código nunca se ejecutó en la sesión de construcción | `--preflight`, `--smoke` y G0–G4 existen para eso; la Fase A bloquea la Fase B si fallan. |
| Versión de pymoo distinta (API cambia entre 0.6.0 y 0.6.1) | Versión fijada en `uv.lock`; los imports se prueban en `--preflight`. |
| Varios hilos por proceso falsean los tiempos y frenan todo | Un hilo por proceso (sección 5). |
| Algún gate falla a media Fase A | `run_all.sh` se detiene y dice cuál; no se lanza la Fase B. |

**Extensiones diferidas** (C0b, C2, búsqueda aleatoria, más corridas del AG, SHTE): ninguna es necesaria para cumplir los requisitos. Si más adelante se quiere agregar alguna, no hace falta tocar lo ya corrido: la caché y los JSONL se conservan y solo se ejecuta lo nuevo.

**SHTE (extensión diferida).** Clonar `https://github.com/lqnankai/Music-Dataset`, escribir `data.py: shte_index()` (la tonalidad viene en el nombre del archivo; parsear y mapear a `keyIndexMap`) y un `train_paths` equivalente. El resto del código no cambia. Solo si sobra tiempo; no es necesaria para cumplir los requisitos.

---

## 11. Cumplimiento de los requisitos de la parte 2

Matriz contra cada viñeta de las instrucciones del proyecto y de la guía del póster.

| Instrucción | Qué exige | Dónde se cumple | Estado |
|---|---|---|---|
| 1 | Paper revisado por pares, ≤ 5 años, con algoritmo bioinspirado | Sci Rep 2026; SNN con STDP (bioinspirada) y el AG que agregamos | Aprobado por el profesor |
| 1 | Metodología replicable, datos públicos | Código BrainCog público + corales de Bach en music21 | Cumple (H1–H10 documentan lo que el paper omite) |
| 2 | Problema | §3 | Cumple |
| 2 | Variables, función objetivo y restricciones | §3 y §4.3–4.4 | Cumple (el paper no las define; las construimos nosotros y lo decimos) |
| 2 | Algoritmo e hiperparámetros | §3.1, §5.4, `ga_config.json`; hiperparámetros de la SNN en §2 y T4 | Cumple |
| 2 | Protocolo experimental y resultados a replicar (≥ 1 tabla o figura) | §4 y §9 (T1: Fig. 5e, Fig. 7b, Tabla 2, Fig. 8a) | Cumple |
| 3 | Programar o adaptar código y citarlo | Se adapta BrainCog (citar) y `pymoo` (Blank & Deb, 2020); `code/` previo | Cumple; falta citar en el póster |
| 3 | Reproducir el protocolo y comparar con lo reportado | `run_replica.py`, T1, F3, F6 | Cumple, con las salvedades de 4.5 |
| 3 | Si no coincide, documentar causas (detalles omitidos, semillas, versiones de datos, nº de evaluaciones) | T5: H3–H5 (detalles omitidos), G2 (semillas), 40 de ≈408 corales y 20 notas (datos), `fast` vs `full` (evaluaciones) | Cumple |
| 4 | Mejora (ajuste de hiperparámetros, variante, restricciones, multiobjetivo) | NSGA-II que ajusta hiperparámetros con restricción de DPR y dos objetivos | Cumple (tres de las cinco ideas a la vez) |
| 5 | ≥ 30 corridas independientes por configuración | C0: 30 semillas, C1: 30 semillas (`full`); el AG corre 1 vez (aprobado) | Cumple, con la interpretación aprobada |
| 5 | Media, mediana y σ | T2 | Cumple |
| 5 | Comparar réplica, artículo y mejora con Wilcoxon | T3 (a) réplica vs artículo, (b) mejora vs réplica, (c) mejora vs artículo; Holm | Cumple |
| 6 | Documentar experimentos | `manifest.json`, `logs/`, §5.7, `ga_config.json` | Cumple |
| 6 | Comportamiento y efectividad del algoritmo | F1, F2, T3 | Cumple |
| 6 | Qué tan replicable fue y dificultades | T1 (veredictos), T5, `bench.json` | Cumple |
| 7 | Póster: título, integrantes, referencia, problema y formulación, algoritmo y metodología, réplica vs original, mejora y resultados, conclusiones, referencias | §9.1 (mapa a secciones del póster) | Cumple; el póster lo arma Matías con T1, T3, F3, F2, F4 |
| 7 | Guía del póster: ≤ 800 palabras, ≈ 3 gráficos, tablas ≤ 4 × 7, 4–5 referencias, conclusiones visibles | §9 (versiones "póster", F2/F3/F4) | Cumple |
| Extra (no pedido) | C0b, C2, aleatoria, más corridas, SHTE | Diferidos | Fuera del alcance actual |
