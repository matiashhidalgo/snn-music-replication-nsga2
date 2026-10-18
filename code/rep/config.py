"""Configuración común de la réplica (perfiles, espacio de búsqueda, rutas y utilidades de E/S).

Solo usa la biblioteca estándar y numpy: no importa vendor ni torch. Todas las rutas son relativas a neural-music/.
"""
import atexit
import hashlib
import importlib.metadata as _md
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[2]                       # neural-music/
REP = ROOT / "code" / "rep"
BRAINCOG = ROOT / "vendor" / "Brain-Cog"
MUSIC = BRAINCOG / "examples/Knowledge_Representation_and_Reasoning/musicMemory"
PAPER_REF = ROOT / "data" / "paper_reference.json"
BACH_INDEX = REP / "data" / "bach_index.json"


def fake():
    """REP_FAKE_EVAL=1: evaluación simulada (sin vendor, torch ni music21). Solo para el selftest."""
    return os.environ.get("REP_FAKE_EVAL") == "1"


def rdir():
    """Carpeta de resultados (REP_RESULTS_DIR, por defecto 'results'), siempre dentro de neural-music/ si es relativa."""
    p = Path(os.environ.get("REP_RESULTS_DIR", "results"))
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------- perfiles (4.1)
PROFILES = {
    "fast":  dict(n_train=5,  n_ref=10, n_gen=6,  L=6),   # dentro del AG
    "full":  dict(n_train=40, n_ref=10, n_gen=10, L=8),   # verificación y corridas finales
    "smoke": dict(n_train=1,  n_ref=2,  n_gen=1,  L=2),   # prueba de humo
}

# ----------------------------------------------------------------------------- espacio de búsqueda (4.3)
VARS = ["stdp_A", "stdp_tau", "w_init", "thr", "izh_d", "alpha_p"]
SPACE = {                    # clave: (defecto del código, cota baja, cota alta, valor del paper)
    "stdp_A":   (5.0,  1.0, 20.0, None),
    "stdp_tau": (5.0,  2.0, 20.0, None),
    "w_init":   (45.0, 20.0, 80.0, None),
    "thr":      (2,    1,    6,    5),
    "izh_d":    (8.0,  2.0, 30.0, 30.0),
    "alpha_p":  (30.0, 15.0, 60.0, 30.0),
}
DEFAULTS = {k: SPACE[k][0] for k in VARS}
LO = np.array([SPACE[k][1] for k in VARS], float)
HI = np.array([SPACE[k][2] for k in VARS], float)
PAPER_LITERAL = {**DEFAULTS, "izh_d": 30.0, "thr": 5}            # C0b (diferida)

DPR_MIN = 0.90               # restricción: DPR >= 0.90
SEEDS_FINAL = list(range(1, 31))
N_PROCS_MAX = 3
METRICS = ["PC", "PR", "PI", "DPR", "PE", "Pp", "PpR", "EBR", "GC"]


F1_METRICS = ("sed9", "sed_pitch5")      # candidatos para f1 (regla fijada en 4.4)


def f1_metric():
    """f1 del AG: lee `f1_metric` de bench.json (lo fija G2); sin bench.json o con valor raro, 'sed9'."""
    v = (read_json(rdir() / "bench.json") or {}).get("f1_metric")
    return v if v in F1_METRICS else "sed9"


def s_fit(run_id):
    """Semilla fija del fitness del AG (4.2)."""
    return 1000 + int(run_id)


def full_params(p=None):
    """Completa con DEFAULTS; thr se redondea a entero y el resto a float."""
    d = dict(DEFAULTS)
    d.update(p or {})
    return {k: (int(round(d[k])) if k == "thr" else float(d[k])) for k in VARS}


def encode(p):
    p = full_params(p)
    return np.array([p[k] for k in VARS], float)


def decode(x):
    """Vector del AG -> dict de parámetros (dentro de cotas, thr entero)."""
    x = np.clip(np.asarray(x, float), LO, HI)
    return full_params({k: x[i] for i, k in enumerate(VARS)})


# claves que debe traer data/paper_reference.json (sección 9, T1)
REQUIRED_PAPER_KEYS = [
    "cos_pasw_major", "cos_pasw_minor", "cos_psc_major", "cos_psc_minor",
    "psc_G_major", "pasw_G_major", "cos_C_major_psc", "cos_C_major_pasw",
    "metric_names", "metrics_original", "metrics_model", "oa_model", "sed",
]

# ----------------------------------------------------------------------------- hash del código
CODE_FILES = ["config.py", "patch.py", "engine.py", "data.py", "ksprofile.py", "metrics9.py", "evaluate.py"]


def code_hash():
    """sha1 del código que define los resultados: si alguien lo edita, la caché vieja deja de valer."""
    h = hashlib.sha1()
    for f in CODE_FILES:
        p = REP / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.exists() else b"<falta>")
    return h.hexdigest()


# ----------------------------------------------------------------------------- E/S segura
def _conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"no serializable: {type(o)}")


def jdumps(o, **kw):
    return json.dumps(o, default=_conv, **kw)


def write_json_atomic(path, obj):
    """Escribe a .tmp, fsync y os.replace: nunca queda un archivo a medias."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(jdumps(obj, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def append_jsonl(path, obj):
    """Una línea JSON por resultado, con flush + fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(jdumps(obj, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path, repair=True):
    """Lee un JSONL. Si la última línea quedó truncada (corte de luz), se descarta; con repair=True
    también se recorta el archivo para que el siguiente append no la pegue a una línea rota."""
    path = Path(path)
    if not path.exists():
        return []
    out, good_end, pos = [], 0, 0
    with open(path, "rb") as f:
        for raw in f:
            pos += len(raw)
            if not raw.endswith(b"\n"):
                break
            s = raw.strip()
            if not s:
                good_end = pos
                continue
            try:
                out.append(json.loads(s))
                good_end = pos
            except ValueError:
                break
    if repair and good_end != path.stat().st_size:
        with open(path, "r+b") as f:
            f.truncate(good_end)
    return out


def parse_seeds(s):
    """'1-30' | '1,2,5-7' -> lista de enteros."""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def versions():
    v = {"python": platform.python_version()}
    for pkg in ["torch", "music21", "pymoo", "scipy", "numpy", "pandas", "matplotlib", "openpyxl"]:
        try:
            v[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            v[pkg] = None
    return v


def manifest_add(stage, argv=None):
    """Agrega una entrada a results/manifest.json (versiones, máquina, fecha, code_hash, argumentos)."""
    p = rdir() / "manifest.json"
    m = read_json(p, {"entries": []})
    m["entries"].append(dict(
        stage=stage, argv=argv if argv is not None else sys.argv[1:], date=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=socket.gethostname(), machine=platform.machine(), system=platform.platform(),
        versions=versions(), code_hash=code_hash(), fake=fake(),
        results_dir=str(rdir().relative_to(ROOT)) if str(rdir()).startswith(str(ROOT)) else str(rdir())))
    write_json_atomic(p, m)


# ----------------------------------------------------------------------------- guardia de procesos
def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def procs_guard(procs, name="script"):
    """La suma de --procs de lo que corre a la vez no puede pasar de N_PROCS_MAX (~3,3 GB de RAM por proceso)."""
    if procs > N_PROCS_MAX:
        print(f"[{name}] --procs {procs} supera el máximo ({N_PROCS_MAX}). Me detengo.", flush=True)
        sys.exit(2)
    d = rdir() / ".procs"
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in d.glob("*.json"):
        info = read_json(f, {})
        pid = info.get("pid")
        if pid and pid != os.getpid() and _alive(pid):
            total += int(info.get("procs", 0))
        elif not (pid and _alive(pid)):
            try:
                f.unlink()
            except OSError:
                pass
    if total + procs > N_PROCS_MAX:
        print(f"[{name}] ya corren {total} procesos de otra etapa; con --procs {procs} pasaría de {N_PROCS_MAX}. "
              f"Espera a que termine o baja --procs. Me detengo.", flush=True)
        sys.exit(2)
    me = d / f"{os.getpid()}.json"
    write_json_atomic(me, dict(pid=os.getpid(), procs=procs, name=name))
    atexit.register(lambda: me.exists() and me.unlink())
