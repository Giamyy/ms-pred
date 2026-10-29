#!/usr/bin/env python3
"""Train and evaluate MARASON on data/spec_datasets/msg_simulation on a local GPU machine.

Same steps as run_scripts/marason/run_all.sh (without MAGMA, which the dataset already has):

    B1  train fragment generator      configs/marason/marason_train_msg.yaml
    B2  predict DAGs                  configs/marason/marason_gen_predict_train_msg.yaml
    B3  add true intensities          data_scripts/dag/add_dag_intens.py
    B4  references + train intensity  configs/marason/marason_inten_train_msg.yaml
    B5  predict + evaluate test       as run_scripts/marason/05_predict_marason_inten.py
    B6  retrieval                     as run_scripts/marason/run_retrieval_msg.py

The configs are copied to results/marason_msg_simulation/local_configs/ with this machine's
GPUs / workers / batch sizes; the files in configs/ stay unchanged. Each finished step writes
a .done_B* marker, so re-running the script resumes after the last finished step.

    python scripts/train_marason_msg.py                        # all steps
    python scripts/train_marason_msg.py --steps B5 B6          # only some steps
    python scripts/train_marason_msg.py --gpus 0,1 --gen-batch 16
    python scripts/train_marason_msg.py --steps B4 --force     # redo a finished step
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ALL_STEPS = ["B1", "B2", "B3", "B4", "B5", "B6"]
DATASET = "msg_simulation"

REPO = Path(__file__).resolve().parents[1]
DATA = Path("data/spec_datasets") / DATASET
RES = Path("results") / f"marason_{DATASET}"
RUN = RES / "split_rnd1"
REF_DIR = Path("data") / DATASET / "closest_neighbors/infinite"
CFG_DIR = RES / "local_configs"
LOG_DIR = RES / "logs"


def get_args():
    ncpu = os.cpu_count() or 4
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--num-workers", type=int, default=min(ncpu, 16))
    parser.add_argument("--gen-batch", type=int, default=32, help="B1 batch size (config default 32)")
    parser.add_argument("--inten-batch", type=int, default=16, help="B4 batch size (config default 16)")
    parser.add_argument("--gen-pred-batch", type=int, default=8, help="B2 batch size (config default 8)")
    parser.add_argument("--inten-pred-batch", type=int, default=64, help="B5 batch size (05_predict default 64)")
    parser.add_argument("--retrieval-batch", type=int, default=10, help="B6 batch size (run_retrieval default 10)")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="B1/B4 epochs (config default 200; early stopping usually ends sooner)")
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS)
    parser.add_argument("--force", action="store_true", help="run the chosen steps even if already done")
    return parser.parse_args()


class Runner:
    def __init__(self, args):
        self.args = args
        self.gpus = args.gpus
        self.log = None
        # Child processes (the launcher calls `python3`) must use the same environment as this script
        self.env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")

    def sh(self, cmd: list, gpu: bool = False):
        """Run a command, print its output live and append it to the current step log."""
        cmd = [str(c) for c in cmd]
        env = dict(self.env, CUDA_VISIBLE_DEVICES=self.gpus) if gpu else self.env
        line = "$ " + " ".join(cmd)
        print(line, flush=True)
        self.log.write(line + "\n")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for out in proc.stdout:
            print(out, end="", flush=True)
            self.log.write(out)
        if proc.wait() != 0:
            raise RuntimeError(f"Lệnh lỗi (exit {proc.returncode}): {' '.join(cmd)}")

    def local_config(self, src: str, dst: Path, **overrides):
        """Copy a launcher config with this machine's GPUs and the given universal args."""
        cfg = yaml.safe_load(open(src))
        cfg["launcher_args"]["visible_devices"] = [int(g) for g in self.gpus.split(",")]
        for key, value in overrides.items():
            cfg["universal_args"][key.replace("_", "-")] = [value]
        with open(dst, "w") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)
        return dst

    def launch(self, src: str, name: str, **overrides):
        cfg = self.local_config(src, CFG_DIR / name, **overrides)
        self.sh([sys.executable, "launcher_scripts/run_from_config.py", cfg])

    # ----- steps -----

    def b1(self):
        a = self.args
        self.launch("configs/marason/marason_train_msg.yaml", "train_gen.yaml",
                    num_workers=a.num_workers, batch_size=a.gen_batch, max_epochs=a.max_epochs)
        require(RUN / "ckpt/gen/best.ckpt")

    def b2(self):
        a = self.args
        self.launch("configs/marason/marason_gen_predict_train_msg.yaml", "gen_predict.yaml",
                    num_cpu_workers=a.num_workers, num_gpu_workers=min(a.num_workers, 4),
                    batch_size=a.gen_pred_batch)
        require(RUN / "preds_train_100/tree_preds.hdf5")

    def b3(self):
        self.sh([sys.executable, "data_scripts/dag/add_dag_intens.py",
                 "--pred-dag-path", RUN / "preds_train_100/tree_preds.hdf5",
                 "--true-dag-path", DATA / "subformulae/no_subform.hdf5",
                 "--out-dag-path", RUN / "preds_train_100_inten.hdf5",
                 "--num-workers", self.args.num_workers, "--add-raw"])

    def b4(self):
        a = self.args
        # References (RAG neighbours) are computed and saved on the first run, loaded afterwards
        have_ref = (REF_DIR / "train_ref.npy").exists() and (REF_DIR / "db_specs.npz").exists()
        if not have_ref:
            REF_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Chưa có phổ tham chiếu trong {REF_DIR}: tính và lưu trong lần train này")
        self.launch("configs/marason/marason_inten_train_msg.yaml", "train_inten.yaml",
                    num_workers=a.num_workers, batch_size=a.inten_batch, max_epochs=a.max_epochs,
                    load_reference=have_ref, save_reference=not have_ref)
        require(RUN / "ckpt/inten/best.ckpt")

    def b5(self):
        self.sh([sys.executable, "src/ms_pred/marason/predict_inten.py",
                 "--batch-size", self.args.inten_pred_batch,
                 "--dataset-name", DATASET,
                 "--split-name", "split.tsv",
                 "--checkpoint-pth", RUN / "ckpt/inten/best.ckpt",
                 "--save-dir", RUN / "preds",
                 "--gpu",
                 "--num-workers", 0,
                 "--magma-dag-folder", RUN / "preds_train_100/tree_preds.hdf5",
                 "--inten-folder", RUN / "preds_train_100_inten.hdf5",
                 "--subset-datasets", "test_only",
                 "--binned-out",
                 "--add-ref",
                 "--max-ref-count", 3,
                 "--ref-dir", REF_DIR], gpu=True)
        self.sh([sys.executable, "analysis/spec_pred_eval.py",
                 "--binned-pred-file", RUN / "preds/binned_preds.hdf5",
                 "--max-peaks", 100,
                 "--min-inten", 0,
                 "--formula-dir-name", "no_subform.hdf5",
                 "--dataset", DATASET])

    def b6(self):
        ran = False
        for kind in ["formula", "mass"]:
            table = DATA / f"retrieval/cands_df_test_{kind}_256.tsv"
            if not table.exists():
                print(f"Không có {table}, bỏ qua retrieval theo {kind}")
                continue
            out = RUN / f"retrieval_{DATASET}_test_{kind}_256"
            out.mkdir(parents=True, exist_ok=True)
            self.sh([sys.executable, "src/ms_pred/marason/predict_smis.py",
                     "--batch-size", self.args.retrieval_batch,
                     "--dataset-name", DATASET,
                     "--sparse-out",
                     "--sparse-k", 250,
                     "--max-nodes", 250,
                     "--split-name", "split.tsv",
                     "--gen-checkpoint", RUN / "ckpt/gen/best.ckpt",
                     "--inten-checkpoint", RUN / "ckpt/inten/best.ckpt",
                     "--save-dir", out,
                     "--ref-dir", REF_DIR,
                     "--dataset-labels", table,
                     "--num-cpu-workers", min(self.args.num_workers, 4),
                     "--num-gpu-workers", 0,
                     "--magma-dag-folder", RUN / "preds_train_100_inten.hdf5",
                     "--adduct-shift",
                     "--gpu",
                     "--embed-elem-group",
                     "--add-ref", "--max-ref-count", 3,
                     "--binned-out"], gpu=True)
            self.sh([sys.executable, "src/ms_pred/retrieval/retrieval_benchmark.py",
                     "--dataset", DATASET,
                     "--true-labels", DATA / "labels.tsv",
                     "--full-labels", table,
                     "--formula-dir-name", "no_subform.hdf5",
                     "--pred-file", out / "binned_preds.hdf5",
                     "--dist-fn", "entropy",
                     "--binned-pred"])
            ran = True
        if not ran:
            raise RuntimeError(f"Không có bảng retrieval nào trong {DATA / 'retrieval'}")

    def run_step(self, name: str):
        done = RUN / f".done_{name}"
        if done.exists() and not self.args.force:
            print(f"== {name}: đã xong trước đó, bỏ qua (--force để chạy lại)")
            return
        # This step and the later ones that depend on it count as not done until this run succeeds
        for later in ALL_STEPS[ALL_STEPS.index(name):]:
            (RUN / f".done_{later}").unlink(missing_ok=True)
        log_path = LOG_DIR / f"{name}.log"
        print(f"== {name}: bắt đầu {datetime.now():%Y-%m-%d %H:%M:%S}, log: {log_path}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as self.log:
            getattr(self, name.lower())()
        done.touch()
        print(f"== {name}: xong sau {(time.time() - t0) / 60:.0f} phút", flush=True)


def require(path: Path):
    if not path.exists():
        raise RuntimeError(f"Không thấy {path}")


def main():
    args = get_args()
    os.chdir(REPO)
    for f in ["labels.tsv", "spec_files.hdf5", "splits/split.tsv",
              "magma_outputs/magma_tree.hdf5", "subformulae/no_subform.hdf5"]:
        if not (DATA / f).exists():
            sys.exit(f"Thiếu {DATA / f}")
    import torch
    if not torch.cuda.is_available():
        sys.exit("torch không thấy GPU (kiểm tra driver / môi trường: source .venv/bin/activate)")

    for d in [CFG_DIR, LOG_DIR, RUN]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {args.gpus} | workers: {args.num_workers} | batch gen/inten: {args.gen_batch}/{args.inten_batch}"
          f" | epochs: {args.max_epochs}")
    steps = [s for s in ALL_STEPS if s in args.steps]
    print("Các bước:", " ".join(steps))

    runner = Runner(args)
    for step in steps:
        runner.run_step(step)
    print(f"Hoàn tất. Kết quả trong {RUN}, log trong {LOG_DIR}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"LỖI: {e}")
