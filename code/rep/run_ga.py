"""NSGA-II (pymoo) que ajusta los 6 hiperparámetros de la SNN. Reanudable: relanzar el mismo comando reproduce lo hecho
(las evaluaciones salen de la caché y pymoo es determinista con la misma semilla).

  python code/rep/run_ga.py --run-id 1 --pop 12 --hours 12 --profile fast --procs 3
  python code/rep/run_ga.py --run-id 1 --pop 6 --gens 3          # --gens explícito manda sobre --hours

Objetivos: f1 = SED (sed9 o sed_pitch5 según f1_metric de bench.json, regla 4.4; se guarda en ga_config.json), f2 = 1 - coseno(PASW mayor, KS mayor); restricción DPR >= 0.90.
La primera vez escribe results/ga_run{ID}/ga_config.json; al relanzar lo reutiliza e ignora --pop/--gens/--hours.
"""
import argparse
import math
import multiprocessing as mp
import time

import numpy as np

import config
import evaluate
from config import (DEFAULTS, DPR_MIN, HI, LO, PROFILES, VARS, append_jsonl, code_hash, decode, encode, full_params,
                    procs_guard, rdir, read_json, s_fit, write_json_atomic)


def plan_gens(pop, hours, procs, profile):
    """GENS = max(5, floor(N_eval / POP) - 1), N_eval = floor(H * 3600 * P / t_eval_fast) (sección 7).
    Devuelve (gens, nota). Sin bench.json usa 10 y lo avisa."""
    b = read_json(rdir() / "bench.json")
    t = b.get("t_eval_fast") if b else None
    if not t:
        return 10, "sin results/bench.json: GENS = 10 por defecto"
    n_eval = math.floor(hours * 3600 * procs / t)
    return max(5, n_eval // pop - 1), f"t_eval_fast={t:.0f}s -> N_eval={n_eval}"


def nondominated(F):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    return NonDominatedSorting().do(np.asarray(F, float), only_non_dominated_front=True)


def initial_population(run_id, pop):
    """Fila 0 = DEFAULTS (el AG nunca queda peor que el punto de partida del autor); el resto, uniforme dentro de las cotas."""
    return np.vstack([encode(DEFAULTS), np.random.RandomState(run_id).uniform(LO, HI, size=(pop - 1, len(VARS)))])


class Records:
    """Guarda el resultado completo de cada individuo evaluado (clave: parámetros decodificados)."""
    def __init__(self):
        self.by_key = {}
        self.cpu = 0.0                                  # segundos de cómputo acumulados (t_train + t_gen)

    @staticmethod
    def key(x):
        p = decode(x)
        return tuple(p[k] for k in VARS)

    def add(self, x, r):
        self.by_key[self.key(x)] = r

    def get(self, x):
        return self.by_key[self.key(x)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=None, help="generaciones nuevas tras la inicial (manda sobre --hours)")
    ap.add_argument("--hours", type=float, default=12.0, help="horas de reloj para el AG (con bench.json fija GENS)")
    ap.add_argument("--profile", default="fast", choices=sorted(PROFILES))
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    procs_guard(a.procs, "run_ga")

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize

    gdir = rdir() / f"ga_run{a.run_id}"
    gdir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(gdir / "ga_config.json")
    if cfg:
        print(f"[ga] reutilizo {gdir / 'ga_config.json'} (pop={cfg['pop']}, gens={cfg['gens']}); se ignora --pop/--gens/--hours", flush=True)
    else:
        if a.gens is not None:
            gens, note = a.gens, "--gens explícito"
        else:
            gens, note = plan_gens(a.pop, a.hours, a.procs, a.profile)
        cfg = dict(run_id=a.run_id, pop=a.pop, gens=gens, note=note, hours=a.hours, profile=a.profile, procs=a.procs,
                   ga_seed=a.run_id, s_fit=s_fit(a.run_id), bounds={k: [float(LO[i]), float(HI[i])] for i, k in enumerate(VARS)},
                   defaults=full_params(DEFAULTS), dpr_min=DPR_MIN, crossover="SBX(prob=0.9, eta=15)",
                   mutation="PM(eta=20)", f1_metric=config.f1_metric(), code_hash=code_hash(), date=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json_atomic(gdir / "ga_config.json", cfg)
        print(f"[ga] ga_config.json: pop={cfg['pop']} gens={cfg['gens']} ({note}); f1 = {cfg['f1_metric']}", flush=True)
    pop, gens, profile, S = cfg["pop"], cfg["gens"], cfg["profile"], cfg["s_fit"]
    f1m = cfg.get("f1_metric", "sed9")                  # ga_config manda (reanudación idéntica); sin el campo, sed9
    config.manifest_add(f"ga_run{a.run_id}")

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.procs, maxtasksperchild=1)        # global del proceso principal; un hilo por proceso hijo
    recs = Records()
    t0 = time.time()
    hist_path = gdir / "history.jsonl"
    hist_path.write_text("")                            # se reescribe desde cero en cada lanzamiento

    class SNNProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(VARS), n_obj=2, n_ieq_constr=1, xl=LO, xu=HI)

        def _evaluate(self, X, out, *args, **kwargs):
            tasks = [(decode(x), S, profile) for x in X]
            results = pool.map(evaluate.eval_params, tasks, chunksize=1)
            F, G = [], []
            for x, r in zip(X, results):
                recs.add(x, r)
                recs.cpu += r["t_train"] + r["t_gen"]
                f1, f2, g = evaluate.fitness(r, f1m)
                F.append([f1, f2])
                G.append([g])
            out["F"], out["G"] = np.array(F), np.array(G)

    class History(Callback):
        def notify(self, algorithm):
            X, F, G = algorithm.pop.get("X"), algorithm.pop.get("F"), algorithm.pop.get("G")
            feas = (G[:, 0] <= 0).tolist()
            append_jsonl(hist_path, dict(
                gen=int(algorithm.n_gen), n_eval=int(algorithm.evaluator.n_eval), n_feasible=int(sum(feas)),
                X=[decode(x) for x in X], F=F.tolist(), G=G[:, 0].tolist(), feasible=feas,
                dpr=[recs.get(x)["dpr"] for x in X], cpu_seconds=round(recs.cpu, 1)))   # determinista (viene de la caché)
            print(f"[ga] gen {algorithm.n_gen}/{gens + 1}: eval={algorithm.evaluator.n_eval} factibles={sum(feas)}/{len(feas)} "
                  f"mejor f1({f1m})={F[:, 0].min():.3f} mejor f2={F[:, 1].min():.4f} ({time.time() - t0:.0f}s)", flush=True)

    x0 = initial_population(a.run_id, pop)
    algo = NSGA2(pop_size=pop, sampling=x0, crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    try:
        res = minimize(SNNProblem(), algo, ("n_gen", gens + 1), seed=a.run_id, callback=History(), verbose=False)
    finally:
        pool.close()
        pool.join()

    X, F, G = res.pop.get("X"), res.pop.get("F"), res.pop.get("G")
    ok = np.where(G[:, 0] <= 0)[0]
    out = dict(run_id=a.run_id, f1_metric=f1m, feasible=bool(len(ok)), n_pop=int(len(X)), points=[], date=time.strftime("%Y-%m-%d %H:%M:%S"))
    if len(ok):
        front = ok[nondominated(F[ok])]
        seen = set()
        for i in sorted(front, key=lambda i: F[i, 0]):
            p = decode(X[i])
            if tuple(p.values()) in seen:
                continue
            seen.add(tuple(p.values()))
            out["points"].append(dict(params=p, F=F[i].tolist(), dpr=recs.get(X[i])["dpr"], g=float(G[i, 0])))
    else:
        i = int(np.argmin(G[:, 0]))                     # nadie factible: se guarda el de menor violación
        out["best_infeasible"] = dict(params=decode(X[i]), F=F[i].tolist(), dpr=recs.get(X[i])["dpr"], g=float(G[i, 0]))
        print("[ga] AVISO: ningún individuo factible (DPR >= 0.90); pareto.json queda vacío.", flush=True)
    write_json_atomic(gdir / "pareto.json", out)
    print(f"[ga] listo: {len(out['points'])} puntos en el frente ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
