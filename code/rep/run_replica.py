"""Réplica: N semillas de una configuración fija (C0 = código por defecto; C0b = literal del paper, diferida).

  python code/rep/run_replica.py --config defaults --seeds 1-30 --profile full --procs 3

Escribe results/replica/C0.jsonl (una línea = un dict completo de evaluate; reanudable) y, si no existe,
results/control_hist.json (histograma de alturas de los corales de entrenamiento vs KS, sin SNN).
"""
import argparse
import multiprocessing as mp
import time

import config
import evaluate
from config import (DEFAULTS, PAPER_LITERAL, PROFILES, append_jsonl, fake, parse_seeds, procs_guard, rdir, read_jsonl,
                    write_json_atomic)

CONFIGS = {"defaults": ("C0", DEFAULTS), "paper_literal": ("C0b", PAPER_LITERAL)}


def ensure_control_hist(n_train=40):
    p = rdir() / "control_hist.json"
    if p.exists():
        return
    import ksprofile
    ch = ksprofile.hist_control_fake(n_train) if fake() else ksprofile.hist_control(n_train)
    write_json_atomic(p, ch)
    print(f"control_hist.json escrito (coseno hist-KS mayor {ch['major']['cos_ks']:.3f}, menor {ch['minor']['cos_ks']:.3f})",
          flush=True)


def run_seeds(params, seeds, profile, out, procs, label):
    """Evalúa las semillas que faltan en `out` (JSONL reanudable). Devuelve (ya_existían, nuevas)."""
    done = {r["seed"] for r in read_jsonl(out) if not r.get("error")}      # una evaluación con error se reintenta
    todo = [s for s in seeds if s not in done]
    print(f"[{label}] {len(done & set(seeds))} semillas ya existían; faltan {len(todo)}", flush=True)
    if not todo:
        return len(done & set(seeds)), 0
    t0 = time.time()
    tasks = [(params, s, profile) for s in todo]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(procs, len(tasks)), maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap(evaluate.eval_params, tasks, chunksize=1), 1):
            append_jsonl(out, r)
            print(f"[{label}] {i}/{len(tasks)} semilla {r['seed']}: sed9={r['sed9']:.3f} f2={1 - r['cos_major_pasw']:.4f} "
                  f"dpr={r['dpr']:.3f}{' ERROR' if r['error'] else ''} ({time.time() - t0:.0f}s)", flush=True)
    return len(done & set(seeds)), len(todo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="defaults", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    procs_guard(a.procs, "run_replica")
    label, params = CONFIGS[a.config]
    config.manifest_add(f"replica_{label}")
    ensure_control_hist()
    run_seeds(params, parse_seeds(a.seeds), a.profile, rdir() / "replica" / f"{label}.jsonl", a.procs, label)
    print(f"[{label}] listo", flush=True)


if __name__ == "__main__":
    main()
