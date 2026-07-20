"""KPI accuracy benchmark: calibrate (n_points, n_configs) for model benchmarking.

Three runs over 10 stored checkpoints (see NETWORKS), all through the unified
``evaluate_residual_samples`` path:

  run 1  fix 200 plasma configs; sweep points-per-config x 4 Sobol scramble
         seeds against a 16,384-point reference -> pick n_points*.
  run 2  fix n_points*; sweep config-count x 4 independent config draws
         against the 200-config reference -> pick n_configs*.
  run 3  8 joint (point-seed, config-seed) draws at (n_points*, n_configs*),
         plus x2 escalations of each knob, to confirm the baseline is stable.

Each run appends to data/kpi_accuracy/run<N>.json; ``--analyze`` prints the
summary tables (relative errors vs reference, fused-score ranking stability).

  uv run python scripts/kpi_accuracy_benchmark.py --run 1
  uv run python scripts/kpi_accuracy_benchmark.py --run 2 --n-points 2048
  uv run python scripts/kpi_accuracy_benchmark.py --run 3 --n-points 2048 --n-configs 100
  uv run python scripts/kpi_accuracy_benchmark.py --analyze
"""

import argparse
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from src.engine.model_evaluation import _kpi_sample_points, evaluate_residual_samples
from src.engine.network import BASE_SEED
from src.engine.network_manager import NetworkManager
from src.lib.geometry_config import PlasmaConfig
from src.lib.network_config import DomainBounds, HyperParams
from src.lib.run_artifacts import load_config

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "kpi_accuracy"

# 10 checkpoints spanning median |R_GS| 1.8e-3 .. 0.19 and five architectures,
# including the ~1.85e-3 near-tie cluster that stresses ranking stability.
_MSB = "data/benchmarks/2026_07_1{}_model-selection-benchmark"
NETWORKS = {
    "soft-bc": _MSB.format("3_00_49_05_soft-bc-reproduction"),
    "hard-bc-unopt": _MSB.format("3_01_18_41_hard-bc-unoptimized"),
    "hard-bc-tuned": _MSB.format("3_06_59_27_hard-bc-tuned-schedule"),
    "hard-bc-final": _MSB.format("3_08_32_59_hard-bc-final"),
    "n6-5x200-a": _MSB.format("4_12_36_29_default"),
    "n6-5x200-b": _MSB.format("4_21_23_35_default"),
    "arch-6x256": _MSB.format("5_11_05_23_default"),
    "arch-5x320": (
        "data/hpo/2026_07_16_00_16_02_arch_wide_or_deep_2400ep_4a25aff/pinn_2026_07_16_03_21_00"
    ),
    "ff64-a": "data/benchmarks/2026_07_16_22_07_48_default_ba6ccac",
    "ff64-b": "data/benchmarks/2026_07_16_23_11_02_default_8ba1cbf",
}

POINT_BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192)
REF_POINTS = 16_384
POINT_SEEDS = (BASE_SEED + 124, BASE_SEED + 225, BASE_SEED + 326, BASE_SEED + 427)
CONFIG_BUDGETS = (12, 25, 50, 100, 200)
CONFIG_SEEDS = (BASE_SEED + 7, BASE_SEED + 17, BASE_SEED + 27, BASE_SEED + 37)
REF_CONFIG_SEED = BASE_SEED + 123  # same domain-Sobol stream as the eval CLI
CORE_RHO = 0.85
SCORE_BETA = 0.3  # fused HPO ranking score = median + beta * p95

# Config-chunk per point budget: keeps live point-evals <= ~163k so the widest
# net (5x320, nested jvp) stays well inside 12 GB; divisors of 200 avoid
# remainder-shaped recompiles.
CONFIG_CHUNK = {16_384: 10, 8192: 20, 4096: 40, 2048: 50, 1024: 100}


def load_manager(rel_path: str) -> NetworkManager:
    run_dir = REPO / rel_path
    hp = HyperParams.from_dict(load_config(run_dir))
    manager = NetworkManager(hp)
    manager.state = manager.state.replace(params=manager.from_disk(run_dir / "network.flax"))
    return manager


def make_configs(manager: NetworkManager, n_configs: int, seed: int) -> list[PlasmaConfig]:
    """Plasma configs from a scrambled domain-Sobol draw (network-independent)."""
    lower, upper = DomainBounds.get_bounds()
    sobol = qmc.Sobol(d=len(lower), scramble=True, seed=seed)
    values = jnp.array(
        qmc.scale(sobol.random(n_configs), np.asarray(lower), np.asarray(upper)),
        dtype=jnp.float32,
    )
    inputs = manager.sampler.sample_flux_input(plasma_configs=values)
    return list(inputs.config)


def eval_stats(
    manager: NetworkManager, configs: list[PlasmaConfig], n_points: int, seed: int
) -> dict[str, float]:
    """Pooled |R_GS| stats over configs x points, matching evaluate_plasma_kpis."""
    chunk = CONFIG_CHUNK.get(n_points, 200)
    start = time.perf_counter()
    parts = [
        evaluate_residual_samples(manager, configs[i : i + chunk], sample_size=n_points, seed=seed)
        for i in range(0, len(configs), chunk)
    ]
    loss = np.concatenate(parts)
    elapsed = time.perf_counter() - start
    _, rho = _kpi_sample_points(n_points, seed)
    core = np.asarray(rho) < CORE_RHO
    stats = {
        "loss_median": float(np.median(loss)),
        "loss_mean": float(np.mean(loss)),
        "loss_p95": float(np.percentile(loss, 95)),
        "loss_p05": float(np.percentile(loss, 5)),
        "core_loss_median": float(np.median(loss[:, core])),
        "core_loss_p95": float(np.percentile(loss[:, core], 95)),
        "edge_loss_p95": float(np.percentile(loss[:, ~core], 95)),
        "seconds": elapsed,
    }
    stats["fused_score"] = stats["loss_median"] + SCORE_BETA * stats["loss_p95"]
    return stats


def save(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")


def run1() -> None:
    results: dict[str, dict] = {}
    for label, path in NETWORKS.items():
        manager = load_manager(path)
        configs = make_configs(manager, 200, REF_CONFIG_SEED)
        entry: dict[str, dict] = {"ref": {}, "budgets": {}}
        for seed in POINT_SEEDS:
            entry["ref"][str(seed)] = eval_stats(manager, configs, REF_POINTS, seed)
        for n_points in POINT_BUDGETS:
            entry["budgets"][str(n_points)] = {
                str(seed): eval_stats(manager, configs, n_points, seed) for seed in POINT_SEEDS
            }
        results[label] = entry
        save("run1.json", results)  # crash-safe incremental write
        ref_med = np.mean([entry["ref"][str(s)]["loss_median"] for s in POINT_SEEDS])
        print(f"[run1] {label}: ref median={ref_med:.4e}", flush=True)


def run2(n_points: int) -> None:
    results: dict[str, dict] = {"n_points": n_points, "networks": {}}
    for label, path in NETWORKS.items():
        manager = load_manager(path)
        entry: dict[str, dict] = {}
        for n_configs in CONFIG_BUDGETS:
            entry[str(n_configs)] = {}
            for cseed in CONFIG_SEEDS:
                configs = make_configs(manager, n_configs, cseed)
                entry[str(n_configs)][str(cseed)] = eval_stats(
                    manager, configs, n_points, POINT_SEEDS[0]
                )
        results["networks"][label] = entry
        save("run2.json", results)
        print(f"[run2] {label} done", flush=True)


def run3(n_points: int, n_configs: int) -> None:
    draws = [(BASE_SEED + 500 + i, BASE_SEED + 100 + i) for i in range(8)]
    variants = {
        "baseline": (n_points, n_configs, draws),
        "double_points": (2 * n_points, n_configs, draws[:4]),
        "double_configs": (n_points, 2 * n_configs, draws[:4]),
    }
    results: dict[str, dict] = {"n_points": n_points, "n_configs": n_configs, "networks": {}}
    for label, path in NETWORKS.items():
        manager = load_manager(path)
        entry: dict[str, dict] = {}
        for name, (pts, cfgs, variant_draws) in variants.items():
            entry[name] = {}
            for pseed, cseed in variant_draws:
                configs = make_configs(manager, cfgs, cseed)
                entry[name][f"{pseed}/{cseed}"] = eval_stats(manager, configs, pts, pseed)
        results["networks"][label] = entry
        save("run3.json", results)
        print(f"[run3] {label} done", flush=True)


# --------------------------------------------------------------------------
# Analysis


STATS = ("loss_median", "loss_mean", "loss_p95", "loss_p05", "core_loss_median", "edge_loss_p95")


def _rel_errors(trials: list[dict], ref: dict) -> dict[str, float]:
    """Worst relative deviation from the reference, per statistic."""
    return {s: max(abs(t[s] - ref[s]) / (abs(ref[s]) + 1e-12) for t in trials) for s in STATS}


def _ranking(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=scores.get)


def analyze() -> None:
    if (OUT_DIR / "run1.json").exists():
        run1_data = json.loads((OUT_DIR / "run1.json").read_text())
        print("\n=== Run 1: points per config (200 configs, 4 point-seeds) ===")
        ref_stats = {
            label: {s: float(np.mean([entry["ref"][k][s] for k in entry["ref"]])) for s in STATS}
            | {
                "fused_score": float(
                    np.mean([entry["ref"][k]["fused_score"] for k in entry["ref"]])
                )
            }
            for label, entry in run1_data.items()
        }
        ref_rank = _ranking({k: v["fused_score"] for k, v in ref_stats.items()})
        print(f"reference ranking (fused, 16384 pts): {ref_rank}")
        # Reference self-noise: worst seed deviation from the seed-mean at 16,384.
        self_noise = {
            s: max(
                _rel_errors(list(entry["ref"].values()), ref_stats[label])[s]
                for label, entry in run1_data.items()
            )
            for s in STATS
        }
        print("ref self-noise (worst net):", {s: f"{v:.2%}" for s, v in self_noise.items()})
        for n_points in map(str, POINT_BUDGETS):
            worst = dict.fromkeys(STATS, 0.0)
            rank_flips = 0
            for seed in map(str, POINT_SEEDS):
                scores = {}
                for label, entry in run1_data.items():
                    trial = entry["budgets"][n_points][seed]
                    ref = ref_stats[label]
                    for s in STATS:
                        worst[s] = max(worst[s], abs(trial[s] - ref[s]) / (abs(ref[s]) + 1e-12))
                    scores[label] = trial["fused_score"]
                rank_flips += _ranking(scores) != ref_rank
            secs = np.mean(
                [e["budgets"][n_points][str(POINT_SEEDS[0])]["seconds"] for e in run1_data.values()]
            )
            print(
                f"n_points={n_points:>6}: "
                + " ".join(f"{s}={worst[s]:.2%}" for s in STATS)
                + f" rank_flips={rank_flips}/4 t={secs:.2f}s"
            )

    if (OUT_DIR / "run2.json").exists():
        run2_data = json.loads((OUT_DIR / "run2.json").read_text())
        run1_data = json.loads((OUT_DIR / "run1.json").read_text())
        ref_stats = {
            label: {s: float(np.mean([e["ref"][k][s] for k in e["ref"]])) for s in STATS}
            | {"fused_score": float(np.mean([e["ref"][k]["fused_score"] for k in e["ref"]]))}
            for label, e in run1_data.items()
        }
        ref_rank = _ranking({k: v["fused_score"] for k, v in ref_stats.items()})
        print(f"\n=== Run 2: config count (n_points={run2_data['n_points']}, 4 draws) ===")
        for n_configs in map(str, CONFIG_BUDGETS):
            worst = dict.fromkeys(STATS, 0.0)
            rank_flips = 0
            for cseed in map(str, CONFIG_SEEDS):
                scores = {}
                for label, entry in run2_data["networks"].items():
                    trial = entry[n_configs][cseed]
                    ref = ref_stats[label]
                    for s in STATS:
                        worst[s] = max(worst[s], abs(trial[s] - ref[s]) / (abs(ref[s]) + 1e-12))
                    scores[label] = trial["fused_score"]
                rank_flips += _ranking(scores) != ref_rank
            print(
                f"n_configs={n_configs:>4}: "
                + " ".join(f"{s}={worst[s]:.2%}" for s in STATS)
                + f" rank_flips={rank_flips}/4"
            )

    if (OUT_DIR / "run3.json").exists():
        run3_data = json.loads((OUT_DIR / "run3.json").read_text())
        print(
            f"\n=== Run 3: stability at n_points={run3_data['n_points']}, "
            f"n_configs={run3_data['n_configs']} (independent joint draws) ==="
        )
        for variant in ("baseline", "double_points", "double_configs"):
            spreads = {s: [] for s in STATS}
            rank_sets: list[list[str]] = []
            for draw in next(iter(run3_data["networks"].values()))[variant]:
                scores = {
                    label: entry[variant][draw]["fused_score"]
                    for label, entry in run3_data["networks"].items()
                }
                rank_sets.append(_ranking(scores))
            for entry in run3_data["networks"].values():
                trials = list(entry[variant].values())
                for s in STATS:
                    values = [t[s] for t in trials]
                    spreads[s].append((max(values) - min(values)) / (np.mean(values) + 1e-12))
            stable = all(r == rank_sets[0] for r in rank_sets)
            print(
                f"{variant:>14}: "
                + " ".join(f"{s}={max(v):.2%}" for s, v in spreads.items())
                + f" ranking_stable={stable}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, choices=(1, 2, 3))
    parser.add_argument("--n-points", type=int, help="n_points* from run 1 (runs 2/3)")
    parser.add_argument("--n-configs", type=int, help="n_configs* from run 2 (run 3)")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.run == 1:
        run1()
    elif args.run == 2:
        if not args.n_points:
            parser.error("--run 2 requires --n-points")
        run2(args.n_points)
    elif args.run == 3:
        if not (args.n_points and args.n_configs):
            parser.error("--run 3 requires --n-points and --n-configs")
        run3(args.n_points, args.n_configs)
    if args.analyze or args.run:
        analyze()


if __name__ == "__main__":
    main()
