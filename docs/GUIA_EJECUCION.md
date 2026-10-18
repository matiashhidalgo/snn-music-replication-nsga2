# Guía de ejecución · réplica del paper de música con SNN + algoritmo genético

**Para:** quien corre el código en su Mac (M3/M4, 16 GB). **No hace falta entender el código.** Solo copiar y pegar las órdenes de abajo, en orden, y mandar los archivos que se piden.

**Cuánto tarda:** Fase A ≈ 30–40 min. Fase B ≈ 28 horas (una noche y un día completo), sin que tengas que hacer nada. El Mac tiene que quedarse **enchufado, encendido y con la tapa abierta** todo ese tiempo.

---

## 0. Antes de empezar (5 min)

0. **Comprueba tu Mac:** Apple → Acerca de este Mac. Debe ser un chip **Apple (M1 o superior)** y **macOS 14 (Sonoma) o más reciente**. Con un Mac Intel o un macOS anterior la instalación de `torch` falla. Si no se cumple, avisa a Matías antes de seguir.
1. **Enchufa el Mac** y déjalo así hasta el final.
2. **Que no se suspenda:** Ajustes del Sistema → Batería (o Pantalla) → activa *"Evitar que el Mac se suspenda automáticamente con el adaptador de corriente"*. Desactiva el modo de bajo consumo.
3. **Cierra lo pesado:** Chrome con muchas pestañas, Zoom, juegos, editores de video. El programa usa ≈ 10 GB de RAM.
4. **Espacio en disco:** al menos 15 GB libres (Apple → Acerca de este Mac → Almacenamiento).
5. **Descarga el código.** Matías te dará el enlace del repositorio de GitHub (y, si es privado, te habrá invitado como colaborador: acepta la invitación en tu correo). En el repositorio: botón verde **Code → Download ZIP**. Descomprime el ZIP en tu Escritorio y renombra la carpeta resultante a `neural-music`. (Alternativa: `git clone <enlace> ~/Desktop/neural-music`.)
6. Abre **Terminal** (Cmd + Espacio, escribe "Terminal").
7. Instala `uv` si no lo tienes (copia, pega, Enter; si dice que ya existe, sigue):
   ```bash
   which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Si lo acaba de instalar, **cierra la Terminal y ábrela de nuevo**.
   **Herramientas de desarrollo y Python** (una vez; si ya los tienes, no pasa nada):
   ```bash
   xcode-select -p || xcode-select --install     # compilador; abre una ventana, acepta e instala
   python3 --version                              # debe decir 3.10 o mayor
   ```
   Si `python3 --version` dice 3.9 o menor: `brew install python@3.12` (si no tienes Homebrew, avisa a Matías antes de seguir).
8. Entra a la carpeta:
   ```bash
   cd ~/Desktop/neural-music
   ```
   El código del autor (BrainCog) no viene en el ZIP: la Fase A lo descarga sola (≈ 1.5 MB, fijado a una versión exacta). Por eso hace falta internet y `git` (lo instala `xcode-select --install`).

---

## 1. Fase A: prueba y medición (≈ 30–40 min)

```bash
bash code/rep/run_all.sh A
```

- La primera vez descarga el código del autor y las dependencias (varios minutos, necesita internet), y analiza los corales de Bach (otros varios minutos).
- Al final debe decir **`FASE A OK`**.
- **Mándale a Matías** (AirDrop, Drive, correo): las carpetas `results/`, `results_smoke/` y `logs/`. (Si es pesado, `zip -r faseA.zip results results_smoke logs` y manda el zip.) El archivo más importante es `results_smoke/informe_prueba.md`.
- **Espera su "adelante"** antes de seguir. Esa pausa es a propósito: con los tiempos que midió la Fase A se decide cuánto va a tardar la Fase B y evita desperdiciar un día.

Si dice **`FASE A FALLÓ`**: no sigas. Es normal que la primera vez falle algo: esta fase existe justo para eso y solo cuesta ≈ 10 minutos por intento. Lo único que puedes arreglar tú: repetir `uv sync` si falló una instalación, y cerrar apps o liberar disco si el pre-vuelo lo pide. **Si es un error del programa, no edites nada**: espera a que Matías te mande archivos corregidos y repite `bash code/rep/run_all.sh A`. Manda `logs/`, `results/` y `results_smoke/` a Matías tal cual están.

---

## 2. Fase B: la corrida larga (≈ 28 h)

Con la luz verde de Matías, en la misma carpeta:

```bash
nohup caffeinate -dimsu bash code/rep/run_all.sh B > logs/run_all.log 2>&1 &
```

Eso la deja corriendo en segundo plano: **puedes cerrar la Terminal y no se detiene.** `caffeinate` evita que el Mac se duerma.

**Ver cómo va** (cuando quieras):
```bash
tail -n 20 logs/run_all.log          # qué etapa va (Ctrl+C para salir)
wc -l results/replica/C0.jsonl       # debe llegar a 30
ls results/ga_run1                    # va apareciendo history.jsonl
```

Las etapas son, en orden: `c0` (30 corridas), `ga1` (el algoritmo genético, la más larga), `final` (verificación y 30 corridas finales), `check` y `zip`.

**Mientras corre:** no apagues, no cierres la tapa, no abras programas pesados, no ejecutes nada más del proyecto, no toques `results/`.

---

## 3. Si algo se interrumpe o falla

**Regla de oro: relanza exactamente la misma orden de la Fase B.** Retoma donde se quedó, no pierde lo hecho. **Nunca borres `results/`.**

| Qué pasó | Qué hacer |
|---|---|
| Se fue la luz, el Mac se durmió o lo apagaste | Enchúfalo, ábrelo y repite la orden de la Fase B |
| Cerraste la Terminal y no sabes si sigue | `tail -n 5 logs/run_all.log`; si la hora de la última línea es de hace más de 30 min, repite la orden de la Fase B |
| Aparece `Killed` o `MemoryError` (se acabó la RAM) | Cierra todo lo demás y corre con 2 procesos: `PROCS=2 nohup caffeinate -dimsu bash code/rep/run_all.sh B > logs/run_all.log 2>&1 &` |
| `command not found: uv` | Repite el paso 7 de "Antes de empezar" y abre una Terminal nueva |
| `No space left on device` | Libera espacio (≥ 15 GB) y repite la orden. No borres `neural-music/` |
| Dice `FALLÓ` en alguna etapa | No la repitas a ciegas: manda `logs/` a Matías y espera |
| La Fase B se niega a empezar ("pass_AB") | Falta la Fase A o falló: repite `bash code/rep/run_all.sh A` |
| El Mac se pone muy caliente o va muy lento | Déjalo en una superficie dura y ventilada, sin funda. Si se apaga solo, repite la orden |

---

## 4. Al terminar

1. En el log aparecerá **`TODO COMPLETO`** y se habrá creado **`results_bundle.zip`** en la carpeta `neural-music/`. Si no aparece `TODO COMPLETO`, no mandes nada y avisa a Matías.
2. Mándale **`results_bundle.zip`** (AirDrop, Drive o correo). Pesa menos de 100 MB.
3. **No borres la carpeta `neural-music/`** hasta que Matías confirme que pudo abrir el zip.

---

## 5. Lo que NO hay que hacer

- No edites ningún archivo de código.
- No cambies `PROCS` a más de 3 ni lances dos fases a la vez.
- No borres `results/`, `logs/` ni la carpeta `.venv*`.
- No actualices macOS, Python ni paquetes en medio de la corrida.
