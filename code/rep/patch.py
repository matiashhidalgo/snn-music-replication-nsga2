"""Parametrización del modelo del autor por monkeypatch (vendor/Brain-Cog no se toca).

Con los parámetros por defecto el resultado debe ser EXACTAMENTE el del código original (test_equiv.py).
Hiperparámetros: stdp_A, stdp_tau (STDP tipo 3), w_init (peso inicial clave->nota), thr (umbral de coincidencias
para crear sinapsis inversas), izh_d (parámetro d de Izhikevich) y alpha_p (corriente de entrada de altura).

Nota (leída en el vendor): KeyCluster construye las neuronas de clave con d explícito (2, 4 u 8 según el perfil KS),
así que izh_d cambia las neuronas creadas con el valor por defecto (nota, tempo, modo, acorde, ...) y no las de clave.
"""
import inspect
import math
import random
import textwrap
from types import SimpleNamespace

from Modal import synapse as _syn, izhikevichneuron as _izh
from Areas import cortex as _cx, pac as _pac

DEFAULTS = dict(stdp_A=5.0, stdp_tau=5.0, w_init=45.0, thr=2, izh_d=8.0, alpha_p=30.0)
P = SimpleNamespace(**DEFAULTS)


def _rewrite(func, old, new, module):
    """Reescribe una línea literal del código original (así con los valores por defecto queda idéntico)."""
    src = textwrap.dedent(inspect.getsource(func))
    assert src.count(old) == 1, f"{func.__name__}: '{old}' aparece {src.count(old)} veces (esperaba 1)"
    ns = {}
    exec(src.replace(old, new), module.__dict__, ns)
    return ns[func.__name__]


_orig_cw = _syn.Synapse.computeWeight


def _computeWeight(self, t):                        # solo cambia el tipo 3 (A y tau parametrizados)
    if self.type != 3:
        return _orig_cw(self, t)
    for st in self.pre.spiketime:
        s = t - st - (self.delay - 1) * self.post.timeWindow
        self.weight += (P.stdp_A * math.exp(-s / P.stdp_tau) if s >= 0
                        else -P.stdp_A * math.exp(s / P.stdp_tau))


_orig_stf2 = _syn.Synapse.computeShortTermFacilitation2


def _stf2(self, t):
    sp = self.pre.spiketime
    if self.type == 1:
        if sp and sp[0] + self.delay <= t:
            self.strength = 1
    elif self.type >= 2:
        if sp and sp[0] <= t:
            self.strength = 1


def _connectKey(self, keys, Notes):                 # copia de connectKeyAndNotesUsingKSModel (cortex.py:107)
    noteNeurons = Notes.neurons
    for tone in keys.groups.values():
        tneurons = tone.neurons
        for nindex, noteneu in enumerate(noteNeurons[1:]):
            i = nindex % 12
            syn = _syn.Synapse(tneurons[i], noteneu)
            if tneurons[i].importance < 0:
                syn.excitability = 0
                syn.weight = -100
            else:
                syn.excitability = 1
                syn.weight = (P.w_init - 25) + random.uniform(20, 30)   # defecto: 20 + U(20,30), idéntico
            syn.type = 3
            noteneu.synapses.append(syn)
            noteneu.pre_neurons.append(tneurons[i])


_installed = False


def apply(params=None, fast=False):
    """Instala (una sola vez) los parches y fija los valores de P. Se puede llamar muchas veces."""
    global _installed
    for k, v in {**DEFAULTS, **(params or {})}.items():
        setattr(P, k, int(round(v)) if k == "thr" else float(v))
    _izh.P = P
    _pac.P = P
    if not _installed:
        _syn.Synapse.computeWeight = _computeWeight
        _cx.Cortex.connectKeyAndNotesUsingKSModel = _connectKey
        M = _pac.Music_Sequence_Mem
        M.doConnectToKey = _rewrite(M.doConnectToKey, "temp >= 2", "temp >= P.thr", _pac)
        M.doConnectToMode = _rewrite(M.doConnectToMode, "temp >= 2", "temp >= P.thr", _pac)
        N = _izh.NoteIzhikevichNeuron
        N.computeFilterCurrent = _rewrite(N.computeFilterCurrent, "self.I = 30", "self.I = P.alpha_p", _izh)
        _installed = True
    for cls in vars(_izh).values():                 # d de Izhikevich: se reasigna en CADA apply, antes de crear neuronas
        if isinstance(cls, type) and issubclass(cls, _izh.IzhikevichNeuron) and "__init__" in cls.__dict__:
            d = cls.__init__.__defaults__
            if d and len(d) >= 5:
                cls.__init__.__defaults__ = d[:3] + (P.izh_d,) + d[4:]
    # opcional (GATE 3): equivalente exacto, con spiketime ascendente solo hace falta mirar el primer disparo
    _syn.Synapse.computeShortTermFacilitation2 = _stf2 if fast else _orig_stf2
