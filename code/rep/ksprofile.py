"""Perfiles Krumhansl-Schmuckler, coseno, y PSC/PASW de las sinapsis inversas nota->neurona de modo (H2, H9).

Solo numpy a nivel de módulo: psc_pasw(eng, mode) usa el motor ya entrenado; hist_control() necesita music21.
"""
from pathlib import Path

import numpy as np

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])   # Krumhansl 1990
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def mm(v):
    """Normalización min-max (todo ceros si el rango es 0)."""
    v = np.asarray(v, float)
    r = v.max() - v.min()
    return (v - v.min()) / r if r > 0 else np.zeros_like(v)


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def psc_pasw(eng, mode):
    """mode 0 = mayor, 1 = menor. j = semitonos sobre la tónica.
    PSC = número de sinapsis inversas; PASW = peso medio de esas sinapsis."""
    grp = eng.cortex.pfc.modes.groups[mode]
    psc, wsum = np.zeros(12), np.zeros(12)
    for j, neu in enumerate(grp.neurons):
        for syn in neu.synapses:
            if syn.pre.areaName == "NMSM":          # neurona de altura de la SMS (sinapsis inversa)
                psc[j] += 1
                wsum[j] += syn.weight
    pasw = np.divide(wsum, psc, out=np.zeros(12), where=psc > 0)
    return psc, pasw


def hist_control(n_train=40):
    """Control sin SNN: histograma de clases de altura relativas a la tónica de los corales de entrenamiento
    (primeras 20 notas por voz, como el entrenamiento), por modo, y su coseno con el perfil KS."""
    import core
    core.init_vendor()
    import data
    acc = {0: np.zeros(12), 1: np.zeros(12)}
    cnt = {0: 0, 1: 0}
    for path in data.train_paths(n_train):
        xml = core.m21.corpus.parse(Path(path))
        k = xml.analyze("key")
        idx = core.configs.keyIndexMap.get(str(k))
        if idx is None:
            continue
        mode, tonic = idx % 2, k.tonic.pitchClass
        for part in xml.parts:
            for note in part.flatten().notes[:20]:
                acc[mode][(data._pitch_of(note) - tonic) % 12] += 1
        cnt[mode] += 1
    return _control_dict(n_train, acc, cnt)


def _control_dict(n_train, acc, cnt):
    return dict(
        n_train=n_train,
        major=dict(hist=acc[0].tolist(), n_pieces=cnt[0], cos_ks=cos(mm(acc[0]), mm(KS_MAJOR))),
        minor=dict(hist=acc[1].tolist(), n_pieces=cnt[1], cos_ks=cos(mm(acc[1]), mm(KS_MINOR))))


def hist_control_fake(n_train=40):
    """Versión simulada (selftest): histograma = perfil KS más ruido determinista."""
    rs = np.random.RandomState(7)
    acc = {0: np.round(KS_MAJOR * 100 + rs.normal(0, 20, 12)).clip(0), 1: np.round(KS_MINOR * 100 + rs.normal(0, 20, 12)).clip(0)}
    return _control_dict(n_train, acc, {0: 25, 1: 15})
