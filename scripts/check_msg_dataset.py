#!/usr/bin/env python3
"""Check a processed msg_simulation dataset before training MARASON.

Works on the Kaggle export (msg-simulation / msg-simulation-v2) or on data/spec_datasets/msg_simulation. Needs only
h5py and pandas, so it also runs in a plain Kaggle notebook without installing ms-pred.

    python scripts/check_msg_dataset.py                                  # data/spec_datasets/msg_simulation, else search /kaggle/input
    python scripts/check_msg_dataset.py /kaggle/input/msg-simulation-v2

In a Kaggle notebook cell: paste this file, set DATA_DIR below (or leave None to search /kaggle/input) and run it.
Exit code 1 when an error is found.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import pandas as pd

DATA_DIR = None                  # None = data/spec_datasets/msg_simulation, else the first match under /kaggle/input
EXPECTED_SPECS = 119029          # MassSpecGym 1.5 simulation challenge (README of coleygroup/ms-pred)
MANIFEST = "__predspec_manifest__"

results = []


def report(level, msg):
    results.append(level)
    print(f"[{level}] {msg}", flush=True)


def ok(msg):
    report("OK", msg)


def warn(msg):
    report("CẢNH BÁO", msg)


def error(msg):
    report("LỖI", msg)


def find_data_dir(arg):
    if arg:
        return Path(arg)
    local = Path("data/spec_datasets/msg_simulation")
    if (local / "labels.tsv").exists():
        return local
    cands = [p.parent for p in Path("/kaggle/input").rglob("labels.tsv") if (p.parent / "spec_files.hdf5").exists()]
    cands.sort(key=lambda d: (not d.name.startswith("msg_simulation"), len(d.parts)))
    if not cands:
        sys.exit("Không tìm thấy thư mục có labels.tsv + spec_files.hdf5; truyền đường dẫn làm tham số")
    return cands[0]


def read_json(h5, key):
    raw = h5[key][()]
    if hasattr(raw, "shape") and raw.shape:          # HDF5Dataset.write_str stores a 1-element string array
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def spec_of(key):
    """'MassSpecGymID0000001_collision 30.json' -> 'MassSpecGymID0000001'."""
    return key.rsplit("_collision", 1)[0]


def check_subformula_file(path, label_specs, formulas, name, full_scan):
    """Keys cover every spectrum; optionally parse every entry and summarise the assignments."""
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        specs = {spec_of(k) for k in keys}
        missing = label_specs - specs
        extra = specs - label_specs
        (error if missing else ok)(f"{name}: {len(keys)} mục, phủ {len(label_specs) - len(missing)}/{len(label_specs)} phổ"
                                   + (f"; thiếu ví dụ {sorted(missing)[:3]}" if missing else ""))
        if extra:
            error(f"{name}: {len(extra)} phổ không có trong labels.tsv, ví dụ {sorted(extra)[:3]}")
        bad_names = [k for k in keys if "_collision" not in k or not k.endswith(".json")]
        if bad_names:
            error(f"{name}: {len(bad_names)} khoá sai dạng '<spec>_collision <CE>.json', ví dụ {bad_names[:3]}")

        if not full_scan:
            return set(keys)
        none_tbl, n_peaks, n_formula, wrong_form = 0, [], [], []
        for i, k in enumerate(keys):
            d = read_json(h5, k)
            tbl = d.get("output_tbl")
            if tbl is None:
                none_tbl += 1
            else:
                n_peaks.append(len(tbl.get("mz", [])))
                n_formula.append(sum(1 for f in tbl.get("formula", []) if f))
            spec = spec_of(k)
            if spec in formulas and d.get("cand_form") != formulas[spec]:
                wrong_form.append(k)
            if (i + 1) % 20000 == 0:
                print(f"    đã đọc {i + 1}/{len(keys)} mục của {name}", flush=True)
        frac_none = none_tbl / max(len(keys), 1)
        (warn if frac_none > 0.05 else ok)(
            f"{name}: {none_tbl} mục ({100 * frac_none:.2f}%) không gán được (output_tbl = None)")
        if n_peaks:
            s = pd.Series(n_peaks)
            ok(f"{name}: số đỉnh được gán công thức mỗi phổ: trung vị {s.median():.0f}, trung bình {s.mean():.1f}, "
               f"{(s == 0).sum()} phổ có 0 đỉnh")
        if wrong_form:
            error(f"{name}: {len(wrong_form)} mục có cand_form khác công thức trong labels.tsv, ví dụ {wrong_form[:3]}")
        return set(keys)


def main(data_dir):
    print(f"Kiểm tra dataset: {data_dir}\n")

    required = ["labels.tsv", "spec_files.hdf5", "splits/split.tsv",
                "magma_outputs/magma_tree.hdf5", "subformulae/no_subform.hdf5"]
    optional = ["subformulae/magma_subform_50.hdf5", "retrieval/cands_df_test_formula_256.tsv",
                "retrieval/cands_df_test_mass_256.tsv", "audit.json"]
    for f in required:
        p = data_dir / f
        (ok if p.exists() else error)(f"{f}: " + (f"{p.stat().st_size / 2**20:,.1f} MB" if p.exists() else "THIẾU"))
    for f in optional:
        p = data_dir / f
        (ok if p.exists() else warn)(f"{f}: " + (f"{p.stat().st_size / 2**20:,.1f} MB" if p.exists() else "không có"))
    if not (data_dir / "labels.tsv").exists():
        return

    # labels.tsv
    labels = pd.read_csv(data_dir / "labels.tsv", sep="\t")
    label_specs = set(labels["spec"].astype(str))
    (ok if len(label_specs) == len(labels) else error)(
        f"labels.tsv: {len(labels)} dòng, {len(label_specs)} phổ khác nhau")
    (ok if len(label_specs) == EXPECTED_SPECS else warn)(
        f"labels.tsv: {len(label_specs)} phổ (kỳ vọng {EXPECTED_SPECS} cho msg_simulation)")
    cols = {"spec", "smiles", "formula", "ionization", "collision_energies"}
    (ok if cols <= set(labels.columns) else error)(f"labels.tsv: cột {sorted(cols & set(labels.columns))}"
                                                   f"{'' if cols <= set(labels.columns) else ', thiếu ' + str(sorted(cols - set(labels.columns)))}")
    if "collision_energies" in labels:
        n_unknown = labels["collision_energies"].astype(str).str.contains("nan|imputed").sum()
        (ok if n_unknown == 0 else warn)(f"labels.tsv: {n_unknown} phổ có năng lượng va chạm nan/imputed")
        n_ce = labels["collision_energies"].astype(str).map(lambda s: len(ast.literal_eval(s)) if s.startswith("[") else 1)
        ok(f"labels.tsv: số năng lượng va chạm mỗi phổ: {dict(Counter(n_ce))}")
    formulas = dict(zip(labels["spec"].astype(str), labels["formula"].astype(str))) if "formula" in labels else {}

    # split
    split_specs, test_specs = set(), set()
    if (data_dir / "splits/split.tsv").exists():
        split = pd.read_csv(data_dir / "splits/split.tsv", sep="\t")
        split_specs = set(split["name"].astype(str))
        counts = split["split"].value_counts().to_dict()
        test_specs = set(split.loc[split["split"] == "test", "name"].astype(str))
        (ok if split_specs == label_specs else error)(
            f"splits/split.tsv: {len(split)} dòng, {counts}"
            + ("" if split_specs == label_specs else f"; khác labels.tsv ({len(label_specs ^ split_specs)} phổ)"))

    # spec_files.hdf5
    if (data_dir / "spec_files.hdf5").exists():
        with h5py.File(data_dir / "spec_files.hdf5", "r") as h5:
            spec_keys = {k[:-3] for k in h5.keys() if k.endswith(".ms")}
        missing = label_specs - spec_keys
        (ok if not missing else error)(f"spec_files.hdf5: {len(spec_keys)} phổ; thiếu {len(missing)} phổ của labels.tsv")

    # magma_tree.hdf5
    if (data_dir / "magma_outputs/magma_tree.hdf5").exists():
        with h5py.File(data_dir / "magma_outputs/magma_tree.hdf5", "r") as h5:
            tree_specs = {k for k in h5.keys() if k != MANIFEST}
            has_manifest = MANIFEST in h5
        cov = len(tree_specs & label_specs) / max(len(label_specs), 1)
        (ok if cov >= 0.99 else warn)(f"magma_tree.hdf5: cây MAGMa cho {len(tree_specs & label_specs)}/{len(label_specs)} "
                                      f"phổ ({100 * cov:.2f}%, kỳ vọng ~99.8%)")
        if tree_specs - label_specs:
            error(f"magma_tree.hdf5: {len(tree_specs - label_specs)} phổ không có trong labels.tsv")
        (ok if has_manifest else warn)(f"magma_tree.hdf5: {'có' if has_manifest else 'không có'} manifest {MANIFEST}")

    # subformulae
    no_sub_keys = sub50_keys = None
    if (data_dir / "subformulae/no_subform.hdf5").exists():
        no_sub_keys = check_subformula_file(data_dir / "subformulae/no_subform.hdf5", label_specs, formulas,
                                            "no_subform.hdf5", full_scan=False)
    if (data_dir / "subformulae/magma_subform_50.hdf5").exists():
        sub50_keys = check_subformula_file(data_dir / "subformulae/magma_subform_50.hdf5", label_specs, formulas,
                                           "magma_subform_50.hdf5", full_scan=True)
    if no_sub_keys is not None and sub50_keys is not None:
        (ok if no_sub_keys == sub50_keys else error)(
            "magma_subform_50.hdf5 và no_subform.hdf5 có "
            + ("cùng tập khoá (spec, CE)" if no_sub_keys == sub50_keys
               else f"khoá khác nhau: {len(no_sub_keys - sub50_keys)} chỉ ở no_subform, "
                    f"{len(sub50_keys - no_sub_keys)} chỉ ở magma_subform_50"))

    # retrieval tables
    for kind in ["formula", "mass"]:
        p = data_dir / f"retrieval/cands_df_test_{kind}_256.tsv"
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t", usecols=lambda c: c in {"spec", "smiles", "inchikey"})
        specs = set(df["spec"].astype(str))
        per_spec = df.groupby("spec").size()
        not_label = specs - label_specs
        not_test = specs - test_specs if test_specs else set()
        (ok if not not_label and not not_test else error)(
            f"cands_df_test_{kind}_256.tsv: {len(df)} dòng, {len(specs)} phổ"
            + (f" / {len(test_specs)} phổ test" if test_specs else "")
            + f", ứng viên mỗi phổ: trung vị {per_spec.median():.0f}, tối đa {per_spec.max()}"
            + (f"; {len(not_label)} phổ không có trong labels.tsv" if not_label else "")
            + (f"; {len(not_test)} phổ không thuộc tập test" if not_test else ""))
        if per_spec.max() > 256:
            error(f"cands_df_test_{kind}_256.tsv: có phổ hơn 256 ứng viên")

    n_err, n_warn = results.count("LỖI"), results.count("CẢNH BÁO")
    print(f"\nKết quả: {results.count('OK')} OK, {n_warn} cảnh báo, {n_err} lỗi")
    print("DATASET DÙNG ĐƯỢC CHO MARASON" if n_err == 0 else "CÓ LỖI: xem các dòng [LỖI] ở trên")
    return n_err


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]   # ignore Jupyter's -f <kernel.json>
    in_notebook = "ipykernel" in sys.modules
    n_err = main(find_data_dir(DATA_DIR or (args[0] if args and not in_notebook else None)))
    if n_err and not in_notebook:
        sys.exit(1)
