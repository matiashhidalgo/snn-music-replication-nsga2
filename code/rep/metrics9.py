"""Las 9 métricas del paper (PC, PR, PI, DPR, PE, Pp, PpR, EBR, GC), SED (Ec. 16) y OA (Tabla 2).

El paper no fija las definiciones; las elegidas están en la sección 4.5 del plan. DPR y PE siguen las reglas de
code/metrics.py (escala mayor; menor = natural + 7.º elevado; entropía de Shannon en bits de las clases de altura).
Solo numpy/scipy: no importa music21 ni pretty_midi.

Pieza = 4 voces; cada voz es una lista de (altura MIDI, duración en negras); altura -1 = silencio.
Rejilla de 0.25 negras; compás = 4 negras (16 celdas). Cada métrica se redondea a 2 decimales (como dice el paper).
"""
import math

import numpy as np
from scipy.stats import gaussian_kde

NAMES = ["PC", "PR", "PI", "DPR", "PE", "Pp", "PpR", "EBR", "GC"]
GRID = 0.25
BAR = 16                                       # celdas por compás (4 negras)
_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10, 11]             # natural + 7.º elevado (como code/metrics.py)


def key_info(key_str):
    """'B- major' -> (10, 'major'); 'f# minor' -> (6, 'minor'). ('-' o 'b' = bemol, '#' = sostenido)."""
    name, mode = key_str.split()
    pc = _PC[name[0].lower()] + sum(1 for ch in name[1:] if ch == "#") - sum(1 for ch in name[1:] if ch in "-b")
    return pc % 12, mode


def piece_from_gen(gen_pitches):
    """4 listas de alturas generadas (todas negras) -> pieza."""
    return [[(int(p), 1.0) for p in voice] for voice in gen_pitches]


def piece_metrics(voices, key_str):
    """Vector de 9 métricas (redondeado a 2 decimales) de una pieza en la tonalidad key_str."""
    notes = []                                  # (voz, celda inicial, celda final, altura)
    for v, voice in enumerate(voices):
        t = 0.0
        for p, d in voice:
            s = int(round(t / GRID))
            e = max(s + 1, int(round((t + d) / GRID)))
            if p >= 0:
                notes.append((v, s, e, int(p)))
            t += d
    if not notes:
        return np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 1.0])
    pitches = [n[3] for n in notes]
    total = len(pitches)
    PC = len(set(pitches))
    PR = max(pitches) - min(pitches)
    last, ints = {}, []                         # intervalos entre notas consecutivas dentro de cada voz
    for v, _, _, p in notes:
        if v in last:
            ints.append(abs(p - last[v]))
        last[v] = p
    PI = float(np.mean(ints)) if ints else 0.0
    counts = np.bincount([p % 12 for p in pitches], minlength=12)
    probs = counts[counts > 0] / total
    PE = float(-(probs * np.log2(probs)).sum())
    tonic, mode = key_info(key_str)
    scale = {(tonic + s) % 12 for s in (MAJOR if mode == "major" else MINOR)}
    DPR = sum(1 for p in pitches if p % 12 in scale) / total
    T = max(n[2] for n in notes)
    cells = [set() for _ in range(T)]
    starts = set()
    for _, s, e, p in notes:
        starts.add(s)
        for c in range(s, e):
            cells[c].add(p)
    sound = [len(c) for c in cells]
    on = [x for x in sound if x > 0]
    Pp = float(np.mean(on)) if on else 0.0
    PpR = sum(1 for x in sound if x >= 2) / T
    nb = int(math.ceil(T / 4))
    EBR = sum(1 for b in range(nb) if not any(sound[b * 4:(b + 1) * 4])) / nb
    nbar = int(math.ceil(T / BAR))
    if nbar < 2:
        GC = 1.0
    else:
        vec = [np.array([1 if (b * BAR + k) in starts else 0 for k in range(BAR)]) for b in range(nbar)]
        GC = 1.0 - float(np.mean([np.mean(vec[b] != vec[b + 1]) for b in range(nbar - 1)]))
    return np.round(np.array([PC, PR, PI, DPR, PE, Pp, PpR, EBR, GC], float), 2)


def sed(gen, ref, eps=1e-3):
    """Ec. 16 (con raíz): sqrt(sum_i (mu_gen_i - mu_ref_i)^2 / (sigma_ref_i + eps)^2), sigma con ddof=0."""
    gen, ref = np.asarray(gen, float), np.asarray(ref, float)
    mu_g, mu_r, sd_r = gen.mean(0), ref.mean(0), ref.std(0)
    return float(np.sqrt((((mu_g - mu_r) / (sd_r + eps)) ** 2).sum()))


def sed_shares(gen, ref, eps=1e-3):
    """Contribución de cada una de las 9 métricas a sed9² (fracciones que suman 1; ceros si sed9 = 0) y sigma_ref."""
    gen, ref = np.asarray(gen, float), np.asarray(ref, float)
    sd_r = ref.std(0)
    c = (((gen.mean(0) - ref.mean(0)) / (sd_r + eps)) ** 2)
    return (c / c.sum() if c.sum() > 0 else np.zeros_like(c)), sd_r


def decide_f1_metric(pairs, sd_tol=0.01, share_max=0.5):
    """Regla 4.4 (la aplica G2 sin intervención humana). pairs = [(M_gen, M_ref), ...] de DEFAULTS en `fast`, semillas 1-3.
    Por semilla: parte de sed9² que aportan las métricas con sigma_ref < sd_tol; se promedia entre semillas.
    Si ese promedio supera share_max -> 'sed_pitch5'; si no -> 'sed9'."""
    rows = []
    for g, r in pairs:
        if len(g) == 0 or len(r) == 0:
            continue
        sh, sd = sed_shares(g, r)
        rows.append((sh, sd))
    if not rows:
        return dict(f1_metric="sed9", share_low_sigma=0.0, nota="sin datos: sed9 por defecto", shares_por_metrica=[0.0] * 9,
                    sigma_ref_media=[0.0] * 9, share_por_semilla=[])
    low = [float(sh[sd < sd_tol].sum()) for sh, sd in rows]
    share = float(np.mean(low))
    return dict(f1_metric="sed_pitch5" if share > share_max else "sed9", share_low_sigma=share, share_por_semilla=low,
                shares_por_metrica=np.mean([sh for sh, _ in rows], 0).tolist(), sigma_ref_media=np.mean([sd for _, sd in rows], 0).tolist(),
                regla=f"métricas con sigma_ref < {sd_tol} aportan > {int(share_max * 100)} % de sed9²", nota="")


def sed_pitch5(gen, ref):
    """SED solo con las 5 métricas de altura (PC, PR, PI, DPR, PE)."""
    return sed(np.asarray(gen)[:, :5], np.asarray(ref)[:, :5])


def oa(g, r, n=2000):
    """Overlapping area: integral de min(KDE_gen, KDE_ref). Si una muestra tiene sigma 0: 1 si las medias coinciden, 0 si no."""
    g, r = np.asarray(g, float), np.asarray(r, float)
    if g.std() == 0 or r.std() == 0:
        return 1.0 if abs(g.mean() - r.mean()) < 1e-9 else 0.0
    m = 4 * max(g.std(), r.std()) + 1                         # margen para no cortar las colas de la KDE
    x = np.linspace(min(g.min(), r.min()) - m, max(g.max(), r.max()) + m, n)
    trap = getattr(np, "trapezoid", None) or np.trapz
    try:
        return float(trap(np.minimum(gaussian_kde(g)(x), gaussian_kde(r)(x)), x))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def oa_all(gen, ref):
    gen, ref = np.asarray(gen, float), np.asarray(ref, float)
    return [oa(gen[:, i], ref[:, i]) for i in range(gen.shape[1])]
