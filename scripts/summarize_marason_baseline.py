#!/usr/bin/env python3
"""Collect the results of scripts/run_marason_baseline.py into one table (mean ± std over seeds).

Reads results/marason_<ds>/<run>[_<variant>]/preds/pred_eval.yaml (spectrum similarity on the test fold) and
retrieval_*/rerank_eval_*.yaml (top-k retrieval accuracy). Prints the table and writes, by default,
results/marason_baseline_summary.tsv (mean ± std) and results/marason_baseline_per_seed.tsv (one row per run).

    python scripts/summarize_marason_baseline.py --datasets msg_simulation nplib1 massbank msnlib
    python scripts/summarize_marason_baseline.py --datasets nplib1 --variants marason norag
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
RUN_PATTERN = {"msg_simulation": r"split_rnd(\d+)", "default": r"split_1_rnd(\d+)"}
SPEC_METRICS = ["avg_cos_sim", "avg_entropy_sim", "avg_cos_sim_zero_pep", "avg_coverage"]
TOP_K = [1, 5, 10]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["msg_simulation", "nplib1", "massbank", "msnlib"])
    parser.add_argument("--variants", nargs="+", default=["marason", "norag"])
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="default: <results-dir>/marason_baseline")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    # pred_eval.yaml / rerank_eval_*.yaml also hold every test entry; only the top-level averages are needed
    with open(path) as fp:
        return yaml.safe_load(fp)


def collect(results_dir: Path, dataset: str, variants: list[str]) -> list[dict]:
    rows = []
    pattern = RUN_PATTERN.get(dataset, RUN_PATTERN["default"])
    root = results_dir / f"marason_{dataset}"
    if not root.exists():
        return rows
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = re.fullmatch(pattern + r"(?:_(\w+))?", run_dir.name)
        if not m:
            continue
        seed, variant = int(m.group(1)), m.group(2) or "marason"
        if variant not in variants:
            continue
        row = {"dataset": dataset, "variant": variant, "seed": seed, "run": run_dir.name}
        pred_eval = run_dir / "preds/pred_eval.yaml"
        if pred_eval.exists():
            res = load_yaml(pred_eval)
            for k in SPEC_METRICS:
                row[k.replace("avg_", "")] = res.get(k)
            row["n_test"] = len(res.get("individuals", []))
        for rerank in sorted(run_dir.glob("retrieval_*/rerank_eval_*.yaml")):
            res = load_yaml(rerank)
            tag = rerank.parent.name.replace(f"retrieval_{dataset}_", "")
            for k in TOP_K:
                row[f"{tag}_top{k}"] = res.get(f"avg_top_{k}")
        if len(row) > 4:
            rows.append(row)
    return rows


def main():
    args = get_args()
    rows = [r for ds in args.datasets for r in collect(args.results_dir, ds, args.variants)]
    if not rows:
        raise SystemExit(f"Không tìm thấy kết quả nào trong {args.results_dir}/marason_<dataset>/")
    per_seed = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed"])
    metrics = [c for c in per_seed.columns if c not in ("dataset", "variant", "seed", "run", "n_test")]

    summary = []
    for (ds, var), g in per_seed.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": ds, "variant": var, "seeds": ",".join(map(str, g["seed"]))}
        for col in metrics:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals):
                row[col] = f"{vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}"
        summary.append(row)
    summary = pd.DataFrame(summary)

    out = args.out_prefix or (args.results_dir / "marason_baseline")
    per_seed.to_csv(f"{out}_per_seed.tsv", sep="\t", index=False)
    summary.to_csv(f"{out}_summary.tsv", sep="\t", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("Theo từng lần chạy:\n")
        print(per_seed.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nTrung bình ± độ lệch chuẩn theo seed:\n")
        print(summary.to_string(index=False))
    print(f"\nĐã ghi {out}_per_seed.tsv và {out}_summary.tsv")


if __name__ == "__main__":
    main()
