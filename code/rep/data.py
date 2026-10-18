"""Corales de Bach (music21): índice cacheado, muestreo de referencia y eventos de una pieza.

Orden y filtro iguales a los del wrapper code/c01_music_gen.py: m21.corpus.getComposer('bach') solo .mxl.
Necesita music21 (core.init_vendor); el selftest simulado no usa este módulo.
"""
import functools
import random
import time
from pathlib import Path

import core
from config import BACH_INDEX, read_json, write_json_atomic

core.init_vendor()


@functools.lru_cache(maxsize=1)
def mxl_paths():
    return [str(p) for p in core.m21.corpus.getComposer("bach") if str(p).endswith(".mxl")]


def train_paths(n):
    """Los primeros n .mxl del orden del wrapper (0 = todos)."""
    ps = mxl_paths()
    return ps[:n] if n else ps


def _pitch_of(note):
    """Altura MIDI como en rememberPartNotes: si es acorde, la última nota."""
    if note.isChord:
        p = 0
        for cn in note:
            p = int(cn.pitch.midi)
        return p
    return int(note.pitch.midi)


def _dur_of(note):
    d = float(note.duration.quarterLength)
    return 0.125 if d == 0.0 else d             # igual que el wrapper


def bach_index(rebuild=False):
    """{'entries': [{path, key, mode, n_parts, n_notes, valid, error}]} para todos los .mxl; se parsea UNA vez."""
    paths = mxl_paths()
    idx = None if rebuild else read_json(BACH_INDEX)
    if idx and [e["path"] for e in idx.get("entries", [])] == paths:
        return idx
    m21, entries, t0 = core.m21, [], time.time()
    for i, p in enumerate(paths):
        e = dict(path=p, key=None, mode=None, n_parts=0, n_notes=0, valid=False, error=None)
        try:
            xml = m21.corpus.parse(Path(p))
            e["key"] = str(xml.analyze("key"))
            e["mode"] = e["key"].split()[-1]
            e["n_parts"] = len(xml.parts)
            e["n_notes"] = min((len(part.flat.notes) for part in xml.parts), default=0)
            e["valid"] = bool(e["n_parts"] == 4 and e["key"] in core.configs.keyIndexMap and e["n_notes"] >= 20)
        except Exception as ex:                 # una pieza rara no debe tumbar el índice
            e["error"] = f"{type(ex).__name__}: {ex}"[:200]
        entries.append(e)
        if (i + 1) % 50 == 0:
            print(f"   índice de Bach: {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    idx = dict(version=1, n=len(entries), n_valid=sum(e["valid"] for e in entries), entries=entries)
    write_json_atomic(BACH_INDEX, idx)
    return idx


@functools.lru_cache(maxsize=256)
def _events(path, L):
    xml = core.m21.corpus.parse(Path(path))
    return tuple(tuple((_pitch_of(n), _dur_of(n)) for n in part.flat.notes[:L]) for part in list(xml.parts)[:4])


def ref_sample(seed, n, L):
    """n corales válidos al azar (determinista por semilla): key, chord (primera nota de cada voz) y events
    (primeras L notas por voz con su duración real)."""
    valid = [e for e in bach_index()["entries"] if e["valid"]]
    picks = random.Random(seed).sample(valid, min(n, len(valid)))
    out = []
    for e in picks:
        ev = [list(v) for v in _events(e["path"], L)]
        out.append(dict(path=e["path"], key=e["key"], mode=e["mode"], chord=[v[0][0] for v in ev], events=ev))
    return out


def train_modes(n):
    """(# mayores, # menores) entre los primeros n corales de entrenamiento, según el índice."""
    byp = {e["path"]: e for e in bach_index()["entries"]}
    modes = [byp[p]["mode"] for p in train_paths(n) if p in byp]
    return modes.count("major"), modes.count("minor")
