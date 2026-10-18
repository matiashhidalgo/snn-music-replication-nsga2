"""Motor de la SNN: construir/entrenar (build_engine) y generar una pieza (generate).

core.init_vendor() se llama al importar este módulo (cada proceso hijo 'spawn' debe repetir chdir e imports).
"""
import contextlib
import io
import time
from pathlib import Path

import numpy as np

import core

core.init_vendor()

import patch                                   # noqa: E402  (necesita el vendor en sys.path)
from config import full_params, rdir, read_json   # noqa: E402
from data import train_paths                   # noqa: E402

seed_all = core.seed_all


@contextlib.contextmanager
def _quiet():
    """El vendor imprime mucho ('learning...'): se descarta."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def build_engine(params, n_train, seed, vendor_only=False, fast=None):
    """seed_all -> patch.apply (salvo vendor_only) -> EngineAPI().cortexInit() -> entrena n_train corales
    (rememberMusic + learnFourPartMusic, igual que el wrapper). Devuelve (motor, segundos de entrenamiento).
    Los corales cuya tonalidad no está en keyIndexMap se omiten y se cuentan en eng.n_skipped."""
    p = full_params(params)
    if fast is None:                            # la Fase A decide (GATE 3) si se usa el parche rápido (salidas idénticas)
        fast = bool((read_json(rdir() / "bench.json") or {}).get("use_fast_patch"))
    if not vendor_only:
        patch.apply(p, fast=fast)
    seed_all(seed)
    with _quiet():
        eng = core.EngineAPI()
        eng.cortexInit()
    eng.n_skipped = 0
    t0 = time.time()
    for path in train_paths(n_train):
        xml = core.m21.corpus.parse(Path(path))
        if core.configs.keyIndexMap.get(str(xml.analyze("key"))) is None:
            eng.n_skipped += 1
            continue
        name = Path(path).name
        with _quiet():
            eng.rememberMusic(name, "None")
            eng.learnFourPartMusic(xml, name, "None")
    return eng, time.time() - t0


def generate(eng, key_str, chord, L):
    """generateMelodyWithKey(key, acorde semilla, durations=None, L). El primer acorde ES la semilla, así que se
    generan L-1 acordes nuevos. Devuelve 4 listas de L alturas MIDI; -1 = silencio (paso sin 'N')."""
    key = core.configs.keyIndexMap[key_str]
    seed_chord = np.array([[int(p)] for p in chord])
    with _quiet():
        res = eng.generateMelodyWithKey(key, seed_chord, None, L)
    out = []
    for j in (1, 2, 3, 4):
        pitches = []
        for d in list(res.get(j, []))[:L]:
            n = d.get("N")
            pitches.append(int(n) if n is not None and n >= 0 else -1)
        out.append(pitches + [-1] * (L - len(pitches)))
    return out
