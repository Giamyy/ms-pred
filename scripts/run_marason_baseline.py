#!/usr/bin/env python3
"""Run the original MARASON as a baseline, with several seeds, on msg_simulation, nplib1, massbank and msnlib.

For every dataset and seed this runs the authors' pipeline with their configs (only GPUs, workers, batch sizes,
epochs, seed and output folder are set from the command line):

    B1  train fragment generator      configs/marason[/public]/marason_train_<ds>.yaml
    B2  predict DAGs                  configs/marason[/public]/marason_gen_predict_train_<ds>.yaml
    B3  attach observed spectra       data_scripts/dag/add_dag_intens.py
    B4  references + intensity model  configs/marason[/public]/marason_inten_train_<ds>.yaml
    B5  predict + evaluate test fold  predict_inten.py + analysis/spec_pred_eval.py
    B6  retrieval                     predict_smis.py + retrieval_benchmark.py (datasets with candidate tables)

Variants: "marason" (default) and "norag" (the same model with add-reference false, i.e. the No-RAG ablation of
the paper's Table 2). "norag" reuses B1-B3 of the same seed and writes to <run>_norag.

Results go to results/marason_<ds>/<run>/ with <run> = split_rnd<seed> (msg_simulation) or split_1_rnd<seed>
(public datasets), the folder names of the authors' configs. Reference spectra (RAG neighbours) depend only on the
data, so they are computed by the first intensity run and reused by the other seeds. Every finished step writes a
.done_B* marker, so re-running the same command resumes. Summarize with scripts/summarize_marason_baseline.py.

Data must be in data/spec_datasets/<ds>/ (labels.tsv, spec_files.hdf5, splits/, magma_outputs/,
subformulae/no_subform.hdf5). Missing data is downloaded from Kaggle (--kaggle-dataset <ds>=<owner>/<name>;
msg_simulation defaults to giamyy/msg-simulation-v2) or, for the public datasets, built with
data_scripts/public/build_public_datasets.sh.

    python scripts/run_marason_baseline.py --datasets msg_simulation nplib1 massbank msnlib --seeds 1 2 3
    python scripts/run_marason_baseline.py --datasets nplib1 --seeds 1 2 3 --variants marason norag
    python scripts/run_marason_baseline.py --datasets massbank --kaggle-dataset massbank=<owner>/<name>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = ["B1", "B2", "B3", "B4", "B5", "B6"]
SEED_STEPS = ["B1", "B2", "B3"]          # shared by the variants of one seed
VARIANT_STEPS = ["B4", "B5", "B6"]
REQUIRED = ["labels.tsv", "spec_files.hdf5", "magma_outputs/magma_tree.hdf5", "subformulae/no_subform.hdf5"]


def dataset_profile(name: str) -> dict:
    """Configs, split, run folder and retrieval settings of the authors' scripts for one dataset."""
    if name == "msg_simulation":
        cfg = "configs/marason"
        return {
            "name": name,
            "configs": {"gen": f"{cfg}/marason_train_msg.yaml", "pred": f"{cfg}/marason_gen_predict_train_msg.yaml",
                        "inten": f"{cfg}/marason_inten_train_msg.yaml"},
            "split": "split.tsv",
            "run": "split_rnd{seed}",
            "ref_dir": Path("data/msg_simulation/closest_neighbors/infinite"),
            # as run_scripts/marason/run_retrieval_msg.py
            "retrieval": [{"table": f"cands_df_test_{kind}_256.tsv", "tag": f"test_{kind}_256",
                           "max_nodes": 250, "sparse_k": 250, "batch": 10, "dist": "entropy", "add_ref": True}
                          for kind in ("formula", "mass")],
            "kaggle": "giamyy/msg-simulation-v2",
        }
    if name in ("nplib1", "massbank", "msnlib"):
        cfg = "configs/marason/public"
        return {
            "name": name,
            "configs": {"gen": f"{cfg}/marason_train_{name}.yaml", "pred": f"{cfg}/marason_gen_predict_train_{name}.yaml",
                        "inten": f"{cfg}/marason_inten_train_{name}.yaml"},
            "split": "split_1.tsv",
            "run": "split_1_rnd{seed}",
            "ref_dir": Path(f"data/spec_datasets/{name}/closest_neighbors/split_1"),
            # as run_scripts/marason/public/run_retrieval.py (mirrors run_retrieval_nist.py); only nplib1 ships tables
            "retrieval": [{"table": "cands_df_split_1_50.tsv", "tag": "split_1_50",
                           "max_nodes": 100, "sparse_k": 100, "batch": 8, "dist": "cos", "add_ref": False}],
            "kaggle": None,
        }
    raise ValueError(name)


def get_args():
    ncpu = os.cpu_count() or 4
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, choices=["msg_simulation", "nplib1", "massbank", "msnlib"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--variants", nargs="+", default=["marason"], choices=["marason", "norag"])
    parser.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    parser.add_argument("--force", action="store_true", help="rerun the chosen steps even if done")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--num-workers", type=int, default=min(ncpu, 16))
    parser.add_argument("--gen-batch", type=int, default=32, help="B1 batch size (config default 32)")
    parser.add_argument("--inten-batch", type=int, default=16, help="B4 batch size (config default 16)")
    parser.add_argument("--gen-pred-batch", type=int, default=8, help="B2 batch size (config default 8)")
    parser.add_argument("--inten-pred-batch", type=int, default=64, help="B5 batch size (authors' scripts: 64)")
    parser.add_argument("--max-epochs", type=int, default=200, help="B1/B4 epochs (config default 200, early stopping)")
    parser.add_argument("--retrieval-add-ref", choices=["auto", "yes", "no"], default="auto",
                        help="RAG references in B6 retrieval. auto = as the authors' scripts "
                             "(msg_simulation: yes; nplib1: no, as run_retrieval_nist.py)")
    parser.add_argument("--kaggle-dataset", nargs="*", default=[], metavar="DS=OWNER/NAME",
                        help="Kaggle dataset to download when data/spec_datasets/<ds> is missing")
    parser.add_argument("--skip-download", action="store_true", help="never download or build data")
    return parser.parse_args()


class Runner:
    def __init__(self, args, prof: dict, seed: int):
        self.args, self.prof, self.seed = args, prof, seed
        self.data = Path("data/spec_datasets") / prof["name"]
        self.res = Path("results") / f"marason_{prof['name']}"
        self.run = prof["run"].format(seed=seed)
        self.base = self.res / self.run                       # B1-B3 outputs of this seed
        self.cfg_dir = self.res / "local_configs"
        self.log_dir = self.res / "logs"
        self.log = None
        self.env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")

    # ----- helpers -----

    def vdir(self, variant: str) -> Path:
        return self.base if variant == "marason" else self.res / f"{self.run}_{variant}"

    def sh(self, cmd: list, gpu: bool = False):
        cmd = [str(c) for c in cmd]
        env = dict(self.env, CUDA_VISIBLE_DEVICES=self.args.gpus) if gpu else self.env
        line = "$ " + " ".join(cmd)
        print(line, flush=True)
        self.log.write(line + "\n")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for out in proc.stdout:
            print(out, end="", flush=True)
            self.log.write(out)
        if proc.wait() != 0:
            raise RuntimeError(f"Lệnh lỗi (exit {proc.returncode}): {' '.join(cmd)}")

    def launch(self, kind: str, name: str, iterative: dict, **overrides):
        """Copy an authors' launcher config with this machine's settings and a single run (seed, folder)."""
        cfg = yaml.safe_load(open(self.prof["configs"][kind]))
        cfg["launcher_args"]["visible_devices"] = [int(g) for g in self.args.gpus.split(",")]
        for key, value in overrides.items():
            cfg["universal_args"][key.replace("_", "-")] = [value]
        cfg["iterative_args"] = [{k: [v] for k, v in iterative.items()}]
        self.cfg_dir.mkdir(parents=True, exist_ok=True)
        dst = self.cfg_dir / name
        with open(dst, "w") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)
        self.sh([sys.executable, "launcher_scripts/run_from_config.py", dst])

    # ----- steps -----

    def b1(self, variant):
        a = self.args
        self.launch("gen", f"train_gen_{self.run}.yaml",
                    {"split-name": self.prof["split"], "save-dir": self.run, "seed": self.seed},
                    num_workers=a.num_workers, batch_size=a.gen_batch, max_epochs=a.max_epochs)
        require(self.base / "ckpt/gen/best.ckpt")

    def b2(self, variant):
        a = self.args
        self.launch("pred", f"gen_predict_{self.run}.yaml",
                    {"split-name": self.prof["split"], "checkpoint-pth": str(self.base / "ckpt/gen/best.ckpt"),
                     "save-dir": f"{self.run}/preds_train_100"},
                    num_cpu_workers=a.num_workers, num_gpu_workers=min(a.num_workers, 4), batch_size=a.gen_pred_batch)
        require(self.base / "preds_train_100/tree_preds.hdf5")

    def b3(self, variant):
        self.sh([sys.executable, "data_scripts/dag/add_dag_intens.py",
                 "--pred-dag-path", self.base / "preds_train_100/tree_preds.hdf5",
                 "--true-dag-path", self.data / "subformulae/no_subform.hdf5",
                 "--out-dag-path", self.base / "preds_train_100_inten.hdf5",
                 "--num-workers", self.args.num_workers, "--add-raw"])

    def b4(self, variant):
        a, ref_dir = self.args, self.prof["ref_dir"]
        rag = variant == "marason"
        have_ref = (ref_dir / "train_ref.npy").exists() and (ref_dir / "db_specs.npz").exists()
        if rag and not have_ref:
            ref_dir.mkdir(parents=True, exist_ok=True)
            print(f"Chưa có phổ tham chiếu trong {ref_dir}: tính và lưu trong lần train này")
        self.launch("inten", f"train_inten_{self.vdir(variant).name}.yaml",
                    {"split-name": self.prof["split"], "save-dir": self.vdir(variant).name, "seed": self.seed,
                     "magma-dag-folder": str(self.base / "preds_train_100_inten.hdf5")},
                    num_workers=a.num_workers, batch_size=a.inten_batch, max_epochs=a.max_epochs,
                    add_reference=rag, load_reference=rag and have_ref, save_reference=rag and not have_ref)
        require(self.vdir(variant) / "ckpt/inten/best.ckpt")

    def b5(self, variant):
        out = self.vdir(variant)
        cmd = [sys.executable, "src/ms_pred/marason/predict_inten.py",
               "--batch-size", self.args.inten_pred_batch,
               "--dataset-name", self.prof["name"],
               "--split-name", self.prof["split"],
               "--checkpoint-pth", out / "ckpt/inten/best.ckpt",
               "--save-dir", out / "preds",
               "--gpu",
               "--num-workers", 0,
               "--magma-dag-folder", self.base / "preds_train_100/tree_preds.hdf5",
               "--inten-folder", self.base / "preds_train_100_inten.hdf5",
               "--subset-datasets", "test_only",
               "--binned-out"]
        if variant == "marason":
            cmd += ["--add-ref", "--max-ref-count", 3, "--ref-dir", self.prof["ref_dir"]]
        self.sh(cmd, gpu=True)
        self.sh([sys.executable, "analysis/spec_pred_eval.py",
                 "--binned-pred-file", out / "preds/binned_preds.hdf5",
                 "--max-peaks", 100,
                 "--min-inten", 0,
                 "--formula-dir-name", "no_subform.hdf5",
                 "--dataset", self.prof["name"]])

    def b6(self, variant):
        out, ran = self.vdir(variant), False
        for r in self.prof["retrieval"]:
            table = self.data / "retrieval" / r["table"]
            if not table.exists():
                print(f"Không có {table}, bỏ qua retrieval {r['tag']}")
                continue
            add_ref = variant == "marason" and {"auto": r["add_ref"], "yes": True, "no": False}[self.args.retrieval_add_ref]
            save = out / f"retrieval_{self.prof['name']}_{r['tag']}"
            save.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "src/ms_pred/marason/predict_smis.py",
                   "--batch-size", r["batch"],
                   "--dataset-name", self.prof["name"],
                   "--sparse-out", "--sparse-k", r["sparse_k"], "--max-nodes", r["max_nodes"],
                   "--split-name", self.prof["split"],
                   "--gen-checkpoint", self.base / "ckpt/gen/best.ckpt",
                   "--inten-checkpoint", out / "ckpt/inten/best.ckpt",
                   "--save-dir", save,
                   "--ref-dir", self.prof["ref_dir"],
                   "--dataset-labels", table,
                   "--num-cpu-workers", min(self.args.num_workers, 4),
                   "--num-gpu-workers", 0,
                   "--magma-dag-folder", self.base / "preds_train_100_inten.hdf5",
                   "--adduct-shift", "--gpu", "--embed-elem-group", "--binned-out"]
            if add_ref:
                cmd += ["--add-ref", "--max-ref-count", 3]
            self.sh(cmd, gpu=True)
            self.sh([sys.executable, "src/ms_pred/retrieval/retrieval_benchmark.py",
                     "--dataset", self.prof["name"],
                     "--true-labels", self.data / "labels.tsv",
                     "--full-labels", table,
                     "--formula-dir-name", "no_subform.hdf5",
                     "--pred-file", save / "binned_preds.hdf5",
                     "--dist-fn", r["dist"],
                     "--binned-pred"])
            ran = True
        if not ran:
            print(f"{self.prof['name']}: không có bảng ứng viên, không chạy retrieval")

    # ----- bookkeeping -----

    def marker(self, step, variant):
        return (self.base if step in SEED_STEPS else self.vdir(variant)) / f".done_{step}"

    def invalidate_after(self, step, variant):
        """A rerun step makes the later steps that use its output stale."""
        later = STEPS[STEPS.index(step):]
        # B1-B3 feed every variant of this seed, including ones not run now
        variants = ["marason", "norag"] if step in SEED_STEPS else [variant]
        for s in later:
            for v in variants:
                self.marker(s, v).unlink(missing_ok=True)

    def run_step(self, step, variant):
        tag = f"{self.prof['name']} seed {self.seed} {step}" + ("" if step in SEED_STEPS else f" [{variant}]")
        if self.marker(step, variant).exists() and not self.args.force:
            print(f"== {tag}: đã xong, bỏ qua")
            return
        self.invalidate_after(step, variant)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.marker(step, variant).parent.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{self.vdir(variant).name if step not in SEED_STEPS else self.run}_{step}.log"
        print(f"== {tag}: bắt đầu {datetime.now():%Y-%m-%d %H:%M:%S}, log: {log_path}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as self.log:
            getattr(self, step.lower())(variant)
        self.marker(step, variant).touch()
        print(f"== {tag}: xong sau {(time.time() - t0) / 60:.0f} phút", flush=True)


def require(path: Path):
    if not path.exists():
        raise RuntimeError(f"Không thấy {path}")


def prepare_data(prof: dict, args):
    """Make sure data/spec_datasets/<ds> is complete: download from Kaggle or build the public dataset."""
    data = Path("data/spec_datasets") / prof["name"]
    needed = REQUIRED + [f"splits/{prof['split']}"]
    missing = [f for f in needed if not (data / f).exists()]
    if not missing:
        print(f"== {prof['name']}: dữ liệu đã có trong {data}")
        return
    if args.skip_download:
        raise RuntimeError(f"Thiếu {', '.join(str(data / f) for f in missing)} (đang dùng --skip-download)")
    kaggle = dict(item.split("=", 1) for item in args.kaggle_dataset).get(prof["name"], prof["kaggle"])
    if kaggle:
        import train_marason_msg as tm          # Kaggle download helpers (scripts/train_marason_msg.py)
        tm.DATA, tm.DATASET = data, prof["name"]
        tm.REQUIRED = needed
        dl_args = argparse.Namespace(kaggle_dataset=kaggle, download_dir=Path("data/kaggle_download") / prof["name"],
                                     keep_download=False, skip_download=False, sweep=False)
        tm.prepare_data(dl_args)
    else:
        print(f"== {prof['name']}: dựng dữ liệu bằng data_scripts/public/build_public_datasets.sh (MAGMa có thể mất nhiều giờ)")
        env = dict(os.environ, workers=str(args.num_workers),
                   PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
        subprocess.run(["bash", "data_scripts/public/build_public_datasets.sh", prof["name"]], env=env, check=True)
    missing = [f for f in needed if not (data / f).exists()]
    if missing:
        raise RuntimeError(f"{data} vẫn thiếu {', '.join(missing)}")


def main():
    args = get_args()
    os.chdir(REPO)
    import torch
    if not torch.cuda.is_available():
        sys.exit("torch không thấy GPU (kiểm tra driver / môi trường: source .venv/bin/activate)")
    print(f"Datasets: {args.datasets} | seeds: {args.seeds} | variants: {args.variants} | GPU: {args.gpus} | "
          f"workers: {args.num_workers} | batch gen/inten: {args.gen_batch}/{args.inten_batch} | epochs: {args.max_epochs}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("commit:", commit or "?")

    for name in args.datasets:
        prof = dataset_profile(name)
        prepare_data(prof, args)
        for seed in args.seeds:
            runner = Runner(args, prof, seed)
            for step in [s for s in SEED_STEPS if s in args.steps]:
                runner.run_step(step, "marason")
            for variant in args.variants:
                for step in [s for s in VARIANT_STEPS if s in args.steps]:
                    runner.run_step(step, variant)
    print("Hoàn tất. Tổng hợp: python scripts/summarize_marason_baseline.py --datasets " + " ".join(args.datasets))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"LỖI: {e}")
