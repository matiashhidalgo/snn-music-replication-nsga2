"""evaluate(params, seed, profile) -> dict con TODO lo medido (datos crudos, no resúmenes), con caché en disco.

Una evaluación = entrenar una SNN con `params` y generar piezas (o, con REP_FAKE_EVAL=1, una evaluación simulada
con el mismo esquema, determinista a partir de (params, semilla), que no importa vendor, torch ni music21).

CLI:  python code/rep/evaluate.py --seed 1 --profile smoke [--params '{"thr": 1}'] [--no-cache]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

import config
import metrics9
from config import (DEFAULTS, DPR_MIN, HI, LO, METRICS, PROFILES, VARS, encode, fake, full_params, rdir,
                    read_json, versions, write_json_atomic)
from ksprofile import KS_MAJOR, KS_MINOR, cos, mm

STATS = {"computed": 0, "cached": 0}            # contadores del proceso (los usa el selftest)


# ----------------------------------------------------------------------------- objetivos
def fitness(r, f1_metric="sed9"):
    """(f1, f2, g): minimizar f1 = r[f1_metric] ('sed9' o 'sed_pitch5', regla 4.4) y f2 = 1 - coseno(PASW mayor, KS mayor);
    g <= 0 es factible (DPR >= 0.90)."""
    return r[f1_metric], 1.0 - r["cos_major_pasw"], DPR_MIN - r["dpr"]


# ----------------------------------------------------------------------------- caché
def _rounded(p):
    return {k: (p[k] if k == "thr" else round(float(p[k]), 6)) for k in VARS}


def cache_key(params, seed, profile, chash):
    payload = json.dumps(dict(p=_rounded(full_params(params)), seed=int(seed), profile=profile,
                              code_hash=chash, fake=fake()), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def _cache_path(key):
    return rdir() / "cache" / f"{key}.json"


# ----------------------------------------------------------------------------- evaluación real
def _real(p, seed, cfg):
    import data
    import engine
    from ksprofile import psc_pasw
    eng, t_train = engine.build_engine(p, cfg["n_train"], seed)
    psc_M, pasw_M = psc_pasw(eng, 0)            # ANTES de generar (H8)
    psc_m, pasw_m = psc_pasw(eng, 1)
    refs = data.ref_sample(seed, cfg["n_ref"], cfg["L"])
    t0 = time.time()
    gens = [engine.generate(eng, r["key"], r["chord"], cfg["L"]) for r in refs[:cfg["n_gen"]]]
    t_gen = time.time() - t0
    M_gen = np.array([metrics9.piece_metrics(metrics9.piece_from_gen(g), r["key"]) for g, r in zip(gens, refs)])
    M_ref = np.array([metrics9.piece_metrics(r["events"], r["key"]) for r in refs])
    return dict(t_train=t_train, t_gen=t_gen, psc_major=psc_M, pasw_major=pasw_M, psc_minor=psc_m, pasw_minor=pasw_m,
                M_gen=M_gen, M_ref=M_ref, ref_keys=[r["key"] for r in refs], gen_pitches=gens,
                n_train_skipped=getattr(eng, "n_skipped", 0))


# ----------------------------------------------------------------------------- evaluación simulada (selftest)
_FAKE_KEYS = ["C major", "a minor", "G major", "e minor", "D major", "b minor", "F major", "d minor",
              "B- major", "g minor", "E- major", "c minor", "A major", "f# minor"]


def _seed_int(*parts):
    return int(hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)


def _fake(p, seed, cfg):
    """Funciones suaves de los 6 parámetros. f1 quiere w_init bajo y f2 lo quiere alto: el frente tiene varios
    puntos. El dpr cae por debajo de 0.90 con alpha_p o izh_d altos (a veces viola la restricción)."""
    u = (encode(p) - LO) / (HI - LO)
    uA, uT, uW, uTh, uD, uP = u
    f1 = (8.0 + 60 * (uW - 0.15) ** 2 + 25 * (uA - 0.45) ** 2 + 20 * (uT - 0.35) ** 2
          + 15 * (uTh - 0.20) ** 2 + 18 * (uD - 0.70) ** 2 + 12 * (uP - 0.40) ** 2)
    f2 = (0.02 + 0.50 * (uW - 0.85) ** 2 + 0.20 * (uA - 0.30) ** 2 + 0.15 * (uD - 0.50) ** 2
          + 0.10 * (uT - 0.60) ** 2 + 0.05 * (uP - 0.50) ** 2)
    shift = float(os.environ.get("REP_FAKE_DPR_SHIFT", "0"))
    rs_seed = np.random.RandomState(_seed_int("seed", seed))                      # lo que depende solo de la semilla
    rs_par = np.random.RandomState(_seed_int("par", _rounded(p), seed))           # lo que depende de ambos
    dpr_t = 0.965 - 0.9 * max(0.0, uP - 0.55) - 0.25 * max(0.0, uD - 0.65) + shift + 0.01 * rs_seed.randn()
    n_ref, n_gen, L = cfg["n_ref"], cfg["n_gen"], cfg["L"]
    mu = np.array([8.4, 11.7, 1.92, 0.93, 4.29, 4.19, 1.0, 0.02, 1.0])
    sd = np.array([1.5, 2.0, 0.3, 0.04, 0.3, 0.2, 0.02, 0.02, 0.02])
    M_ref = np.round(np.clip(mu + sd * rs_seed.randn(n_ref, 9), 0, None), 2)
    sign = np.array([1, -1, 1, 0, -1, 1, -1, 0, 0], float)
    delta = (f1 * (1 + 0.02 * rs_par.randn()) / 3.0) * sign
    base = M_ref.mean(0) + delta * M_ref.std(0)
    M_gen = np.round(np.clip(base + 0.3 * sd * rs_par.randn(n_gen, 9), 0, None), 2)
    M_gen[:, 3] = np.round(np.clip(dpr_t + 0.01 * rs_par.randn(n_gen), 0, 1), 2)
    lvl = np.sqrt(f2) * 1.5

    def noisy(ks):                                                                 # PASW/PSC: KS + ruido que sube con f2
        return np.maximum(mm(ks) + lvl * rs_seed.randn(12), 0.0)
    pasw_M, pasw_m = 10 * noisy(KS_MAJOR) + 1, 10 * noisy(KS_MINOR) + 1
    psc_M = np.round(40 * noisy(KS_MAJOR) + 5)
    psc_m = np.round(40 * noisy(KS_MINOR) + 5)
    gen_pitches = rs_par.randint(48, 80, size=(n_gen, 4, L)).tolist()
    ref_keys = [_FAKE_KEYS[i] for i in rs_seed.randint(0, len(_FAKE_KEYS), n_ref)]
    return dict(t_train=0.0, t_gen=0.0, psc_major=psc_M, pasw_major=pasw_M, psc_minor=psc_m, pasw_minor=pasw_m,
                M_gen=M_gen, M_ref=M_ref, ref_keys=ref_keys, gen_pitches=gen_pitches, n_train_skipped=0)


# ----------------------------------------------------------------------------- API
def _assemble(raw, params, seed, profile, chash):
    M_gen, M_ref = np.asarray(raw["M_gen"], float), np.asarray(raw["M_ref"], float)
    cos_ = lambda v, ks: cos(mm(v), mm(ks))         # noqa: E731
    mg, mr = M_gen.mean(0), M_ref.mean(0)
    r = dict(seed=int(seed), profile=profile, params=_rounded(params), code_hash=chash, versions=versions(),
             fake=fake(), t_train=raw["t_train"], t_gen=raw["t_gen"],
             sed9=metrics9.sed(M_gen, M_ref), sed_pitch5=metrics9.sed_pitch5(M_gen, M_ref),
             oa=metrics9.oa_all(M_gen, M_ref), dpr=float(mg[METRICS.index("DPR")]),
             cos_major_pasw=cos_(raw["pasw_major"], KS_MAJOR), cos_minor_pasw=cos_(raw["pasw_minor"], KS_MINOR),
             cos_major_psc=cos_(raw["psc_major"], KS_MAJOR), cos_minor_psc=cos_(raw["psc_minor"], KS_MINOR),
             psc_major=raw["psc_major"], psc_minor=raw["psc_minor"],
             pasw_major=raw["pasw_major"], pasw_minor=raw["pasw_minor"],
             M_gen=M_gen, M_ref=M_ref, ref_keys=raw["ref_keys"], gen_pitches=raw["gen_pitches"],
             metrics_gen_mean=mg, metrics_ref_mean=mr, n_train_skipped=raw.get("n_train_skipped", 0), error=None)
    return json.loads(config.jdumps(r))             # todo a tipos JSON (listas, floats)


def _failure(params, seed, profile, chash, err):
    """Evaluación fallida (4.4): f1 = 1e6, f2 = 1, dpr = 0; se registra el error y el AG sigue."""
    return dict(seed=int(seed), profile=profile, params=_rounded(params), code_hash=chash, versions=versions(),
                fake=fake(), t_train=0.0, t_gen=0.0, sed9=1e6, sed_pitch5=1e6, oa=[0.0] * 9, dpr=0.0,
                cos_major_pasw=0.0, cos_minor_pasw=0.0, cos_major_psc=0.0, cos_minor_psc=0.0,
                psc_major=[0.0] * 12, psc_minor=[0.0] * 12, pasw_major=[0.0] * 12, pasw_minor=[0.0] * 12,
                M_gen=[], M_ref=[], ref_keys=[], gen_pitches=[], metrics_gen_mean=[0.0] * 9,
                metrics_ref_mean=[0.0] * 9, n_train_skipped=0, error=err)


def evaluate(params, seed, profile, use_cache=True):
    p = full_params(params)
    chash = config.code_hash()
    key = cache_key(p, seed, profile, chash)
    cp = _cache_path(key)
    if use_cache:
        hit = read_json(cp)
        if hit is not None:
            STATS["cached"] += 1
            return hit
    cfg = PROFILES[profile]
    try:
        raw = _fake(p, seed, cfg) if fake() else _real(p, seed, cfg)
        r = _assemble(raw, p, seed, profile, chash)
    except Exception:                                # una evaluación rota no tumba al AG
        return _failure(p, seed, profile, chash, traceback.format_exc()[-2000:])
    STATS["computed"] += 1
    if use_cache:
        write_json_atomic(cp, r)
    return r


def eval_params(args):
    """Para multiprocessing (spawn): args = (params, seed, profile)."""
    params, seed, profile = args
    return evaluate(params, seed, profile)


def main():
    ap = argparse.ArgumentParser(description="Una evaluación (entrenar la SNN con params y generar piezas).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    ap.add_argument("--params", default="{}", help='JSON, p. ej. \'{"thr": 1, "izh_d": 30}\'; el resto = DEFAULTS')
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    r = evaluate(json.loads(a.params), a.seed, a.profile, use_cache=not a.no_cache)
    print(json.dumps(dict(seed=r["seed"], profile=r["profile"], params=r["params"], sed9=r["sed9"],
                          f2=1 - r["cos_major_pasw"], dpr=r["dpr"], error=r["error"], seg=round(time.time() - t0, 1))))
    sys.exit(1 if r["error"] else 0)


if __name__ == "__main__":
    main()
