"""Arranque del vendor (BrainCog / musicMemory) sin modificarlo.

init_vendor() hace lo mismo que code/c01_music_gen.py: sys.path, chdir a .../musicMemory/task (el código del autor
usa rutas relativas ../inputs/...), shim de music21 e imports. Se llama al importar engine.py: en macOS
multiprocessing usa 'spawn' y cada proceso hijo debe repetir el chdir y los imports.
Con REP_FAKE_EVAL=1 no hace nada (el selftest no importa vendor, torch ni music21).
"""
import os
import random
import sys
import warnings

from config import BRAINCOG, MUSIC, fake

configs = None        # conf.conf.configs del vendor
EngineAPI = None      # api.music_engine_api.EngineAPI
m21 = None            # music21
torch = None
_done = False


def init_vendor():
    global configs, EngineAPI, m21, torch, _done
    if _done:
        return
    if fake():
        _done = True
        return
    for p in (str(MUSIC), str(BRAINCOG)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(MUSIC / "task")                         # rutas relativas del autor
    warnings.filterwarnings("ignore")
    import torch as _torch
    torch = _torch
    try:                                             # un hilo por proceso (los tiempos del benchmark lo exigen)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import music21 as _m21
    m21 = _m21
    # Shim de compatibilidad (el vendor se escribió para un music21 viejo): Pitch(<Pitch>) ya no se acepta.
    _orig = m21.pitch.Pitch.__init__

    def _pitch_init(self, name=None, **kw):
        if isinstance(name, m21.pitch.Pitch):
            name = name.ps
        _orig(self, name, **kw)

    m21.pitch.Pitch.__init__ = _pitch_init
    # music21 ≥ 10 quitó Stream.flat (desde la v7 era solo un alias de flatten()); el vendor lo usa en rememberPartNotes.
    if not hasattr(m21.stream.Stream, "flat"):
        m21.stream.Stream.flat = property(lambda self: self.flatten())
    from conf.conf import configs as _configs
    from api.music_engine_api import EngineAPI as _EngineAPI
    configs, EngineAPI = _configs, _EngineAPI
    _done = True


def seed_all(s):
    """Fija todas las fuentes de azar: random, numpy y torch."""
    import numpy as np
    random.seed(s)
    np.random.seed(s)
    if torch is not None:
        torch.manual_seed(s)
