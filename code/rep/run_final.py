"""Del frente de Pareto a la configuración final: verificar candidatos en `full`, elegir C1 y correr 30 semillas.

  python code/rep/run_final.py --ga-run 1 --verify 5 --seeds 1-30 --profile full --procs 3
  python code/rep/run_final.py --ga-run 1 --pick min_sed      # C2 (diferida): el factible de menor f1

1. Candidatos = punto de rodilla (mínima distancia al ideal con objetivos min-max) + extremo de mínimo f1
   + puntos repartidos por f1 (máximo --verify).
2. Se evalúan los candidatos Y DEFAULTS en `full` con S_FIT -> results/final/verify.jsonl.
3. C1 = el factible (DPR >= 0.90 en full) de menor distancia normalizada al ideal; empate o ninguno factible:
   el de menor f1. Si no mejora a DEFAULTS en `full`, igual se corre y se informa -> results/final/C1.json.
4. 30 semillas de C1 en `full` -> results/final/C1.jsonl (reanudable).
"""
import argparse
import multiprocessing as mp
import time

import numpy as np

import config
import evaluate
from config import (DEFAULTS, DPR_MIN, PROFILES, VARS, full_params, parse_seeds, procs_guard, rdir, read_json,
                    read_jsonl, s_fit, write_json_atomic, append_jsonl)
from run_replica import run_seeds


def norm_dist(F):
    """Distancia al ideal (0,0) con objetivos min-max normalizados (rango 0 -> 0)."""
    F = np.asarray(F, float)
    r = F.max(0) - F.min(0)
    Z = np.divide(F - F.min(0), r, out=np.zeros_like(F), where=r > 0)
    return np.sqrt((Z ** 2).sum(1))


def pick_candidates(points, n):
    """Rodilla + mínimo f1 + puntos repartidos por f1 (únicos, los que existan, máximo n). -> lista de (etiqueta, params)."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p["F"][0])
    F = [p["F"] for p in pts]
    out, used = [], set()

    def add(label, i):
        key = tuple(pts[i]["params"][k] for k in VARS)
        if key not in used and len(out) < n:
            used.add(key)
            out.append((label, pts[i]["params"]))
    add("rodilla", int(np.argmin(norm_dist(F))))
    add("min_f1", 0)
    for q in (0.25, 0.5, 0.75, 0.9, 0.1):
        add(f"f1_q{int(q * 100)}", int(round(q * (len(pts) - 1))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ga-run", type=int, default=1)
    ap.add_argument("--verify", type=int, default=5, help="candidatos a verificar en full (máx. 5)")
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--pick", default="knee", choices=["knee", "min_sed"], help="knee = C1 (obligatoria); min_sed = C2 (diferida)")
    a = ap.parse_args()
    procs_guard(a.procs, "run_final")
    name = "C1" if a.pick == "knee" else "C2"
    fdir = rdir() / "final"
    fdir.mkdir(parents=True, exist_ok=True)
    par = read_json(rdir() / f"ga_run{a.ga_run}" / "pareto.json")
    if par is None:
        raise SystemExit(f"falta {rdir() / f'ga_run{a.ga_run}' / 'pareto.json'}: corre antes run_ga.py")
    config.manifest_add(f"final_{name}")
    S = s_fit(a.ga_run)
    f1m = par.get("f1_metric") or (read_json(rdir() / f"ga_run{a.ga_run}" / "ga_config.json") or {}).get("f1_metric", "sed9")
    print(f"[final] f1 del AG = {f1m}", flush=True)

    cands = pick_candidates(par["points"], min(a.verify, 5))
    if not par["points"]:
        bi = par.get("best_infeasible")
        print("[final] AVISO: el AG no dejó puntos factibles; uso el de menor violación y lo informo.", flush=True)
        cands = [("sin_factibles", bi["params"])] if bi else []
    if not cands:
        raise SystemExit("[final] no hay candidatos")
    tasks = [("DEFAULTS", full_params(DEFAULTS))] + [(lab, full_params(p)) for lab, p in cands]

    # --- verificación en `full` (reanudable por etiqueta)
    vpath = fdir / "verify.jsonl"
    have = {r["label"]: r for r in read_jsonl(vpath) if not r["result"].get("error")}
    todo = [t for t in tasks if t[0] not in have or have[t[0]]["params"] != t[1]]
    print(f"[final] verificación en {a.profile}: {len(tasks) - len(todo)} ya existían, faltan {len(todo)}", flush=True)
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(a.procs, len(todo)), maxtasksperchild=1) as pool:
            res = pool.imap(evaluate.eval_params, [(p, S, a.profile) for _, p in todo], chunksize=1)
            for (lab, p), r in zip(todo, res):
                rec = dict(label=lab, params=p, s_fit=S, profile=a.profile, result=r)
                append_jsonl(vpath, rec)
                have[lab] = rec
                print(f"[final] {lab}: sed9={r['sed9']:.3f} sed_pitch5={r['sed_pitch5']:.3f} f2={1 - r['cos_major_pasw']:.4f} dpr={r['dpr']:.3f}", flush=True)

    # --- elección de C1 (o C2)
    ver = [have[lab] for lab, _ in tasks if lab in have]
    d0 = next(r for r in ver if r["label"] == "DEFAULTS")["result"]
    elig = [r for r in ver if r["label"] != "DEFAULTS"]
    fz = lambda r: np.array([r["result"][f1m], 1 - r["result"]["cos_major_pasw"]])      # noqa: E731
    feas = [r for r in elig if r["result"]["dpr"] >= DPR_MIN and not r["result"].get("error")]
    note = ""
    if a.pick == "min_sed":
        pool_ = feas or elig
        chosen = min(pool_, key=lambda r: r["result"][f1m])
    elif feas:
        d = norm_dist([fz(r) for r in feas])
        best = np.where(np.isclose(d, d.min(), atol=1e-12))[0]
        chosen = feas[int(best[0])] if len(best) == 1 else min((feas[i] for i in best), key=lambda r: r["result"][f1m])
    else:
        chosen = min(elig, key=lambda r: r["result"][f1m])
        note = "ningún candidato factible en full: se eligió el de menor f1"
    c = chosen["result"]
    improves = bool((c[f1m] < d0[f1m] and (1 - c["cos_major_pasw"]) <= (1 - d0["cos_major_pasw"]))
                    or ((1 - c["cos_major_pasw"]) < (1 - d0["cos_major_pasw"]) and c[f1m] <= d0[f1m]))
    info = dict(name=name, pick=a.pick, label=chosen["label"], params=chosen["params"], feasible_in_full=bool(c["dpr"] >= DPR_MIN),
                f1_metric=f1m, f1=c[f1m], f2=1 - c["cos_major_pasw"], dpr=c["dpr"],
                sed9=c["sed9"], sed_pitch5=c["sed_pitch5"],
                defaults=dict(f1=d0[f1m], f2=1 - d0["cos_major_pasw"], dpr=d0["dpr"], sed9=d0["sed9"], sed_pitch5=d0["sed_pitch5"]),
                domina_a_defaults=improves, note=note, s_fit=S, ga_run=a.ga_run, n_candidatos=len(elig))
    if not improves:
        info["aviso"] = "el AG no mejoró a DEFAULTS en `full` (no lo domina); igual se corren las 30 semillas y se informa"
        print("[final] AVISO: " + info["aviso"], flush=True)
    write_json_atomic(fdir / f"{name}.json", info)
    print(f"[final] {name} = {chosen['label']}: f1={info['f1']:.3f} f2={info['f2']:.4f} dpr={info['dpr']:.3f}", flush=True)

    # --- 30 semillas
    run_seeds(chosen["params"], parse_seeds(a.seeds), a.profile, fdir / f"{name}.jsonl", a.procs, name)
    print(f"[final] listo {name}", flush=True)


if __name__ == "__main__":
    main()
