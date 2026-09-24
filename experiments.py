"""Run the experiments reported in the paper and write its tables and figures.

    python experiments.py --synthetic
    python experiments.py --data "data/Toddler Autism dataset July 2018.csv"

Outputs go to paper/generated/ (LaTeX tables, macros, CSVs) and paper/figures/.
Experiments:
  E1  all models on one split (seed 42): confusion matrices, ROC, Wilson CIs,
      PPV/NPV/likelihood ratios, exact McNemar tests against logistic regression
  E2  all models over repeated stratified 80/20 splits (default 20)
  E3  feature-subset ablation (Q-CHAT items / demographics / all) over the same splits
  E4  permutation importance (LR, RF, CNN) over the first 5 splits, 30 permutations each,
      plus logistic-regression coefficients
  E5  CNN re-trained from 10 random initialisations on the E1 split
"""

from __future__ import annotations

import argparse
import json
import os
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from asd.data import (
    BINARY_COLS, CATEGORICAL_COLS, ITEM_COLS, NUMERIC_COLS, build_preprocessor, clean, load_csv, make_synthetic,
)
from asd.evaluate import metrics, plot_confusion_matrices, plot_roc
from asd.models import build_cnn, cnn_predict_proba, ml_models, train_cnn

SUBSETS = {
    "Q-CHAT-10 items only": ITEM_COLS,
    "Demographics only": NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS,
    "All features": ITEM_COLS + NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS,
}
ALL_COLS = SUBSETS["All features"]
PRETTY = {"Who completed the test": "Respondent", "Family_mem_with_ASD": "Family ASD", "Age_Mons": "Age"}
SHORT = {"Logistic Regression": "LR", "SVM (RBF)": "SVM", "Decision Tree": "DT", "Random Forest": "RF",
         "Gradient Boosting": "GB", "Naive Bayes": "NB", "KNN": "KNN", "1D CNN": "1D CNN"}
POP_PREVALENCE = 0.03


# ---------------------------------------------------------------- helpers
def subset_preprocessor(cols):
    parts = []
    if any(c in ITEM_COLS + BINARY_COLS for c in cols):
        parts.append(("passthrough", "passthrough", [c for c in cols if c in ITEM_COLS + BINARY_COLS]))
    for name, tr, tcols in build_preprocessor().transformers:
        if name != "passthrough" and any(c in cols for c in tcols):
            parts.append((name, clone(tr), [c for c in tcols if c in cols]))
    return ColumnTransformer(parts)


def split_encode(X, y, cols, seed, test_size):
    X_tr, X_te, y_tr, y_te = train_test_split(X[cols], y, test_size=test_size, stratify=y, random_state=seed)
    pre = subset_preprocessor(cols)
    return pre.fit_transform(X_tr).astype("float32"), pre.transform(X_te).astype("float32"), y_tr, y_te, pre, X_te


def fit_models(X_tr, X_te, y_tr, seed, epochs, only=None):
    """Fit every model (or the subset `only`); return test probabilities, fitted models and CNN history."""
    probas, fitted, hist = {}, {}, None
    for name, model in ml_models(seed).items():
        if only and name not in only:
            continue
        fitted[name] = model.fit(X_tr, y_tr)
        probas[name] = model.predict_proba(X_te)[:, 1]
    if not only or "1D CNN" in only:
        cnn, hist = train_cnn(X_tr, y_tr, epochs=epochs, seed=seed)
        fitted["1D CNN"] = cnn
        probas["1D CNN"] = cnn_predict_proba(cnn, X_te)
    return probas, fitted, hist


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def ppv_at(sens: float, spec: float, prev: float) -> float:
    return sens * prev / (sens * prev + (1 - spec) * (1 - prev))


def f3(v: float) -> str:
    return f"{v:.3f}"


def ci(lo: float, hi: float) -> str:
    return f"({lo:.3f}--{hi:.3f})"


def write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def predictor(model):
    if hasattr(model, "predict_proba"):
        return lambda Z: model.predict_proba(Z)[:, 1]
    return lambda Z: np.asarray(model(Z[..., np.newaxis], training=False)).ravel()


def permutation_importance(fn, pre, X_te_raw, y_te, n_perm, rng):
    base = metrics(y_te, fn(pre.transform(X_te_raw).astype("float32")))["roc_auc"]
    out = {}
    for col in ALL_COLS:  # one-hot columns of a variable are permuted together
        drops = []
        for _ in range(n_perm):
            Z = X_te_raw.copy()
            Z[col] = rng.permutation(Z[col].to_numpy())
            drops.append(base - metrics(y_te, fn(pre.transform(Z).astype("float32")))["roc_auc"])
        out[col] = drops
    return out


# ---------------------------------------------------------------- figures
def plot_score_hist(X, y, path: str) -> None:
    score = X[ITEM_COLS].sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.0))
    bins = np.arange(-0.5, 11.5, 1)
    ax.hist([score[y == 0], score[y == 1]], bins=bins, stacked=True, color=["#2c7fb8", "#c0392b"],
            label=["No ASD traits", "ASD traits"])
    ax.axvline(3.5, ls="--", color="k", lw=1)
    ax.text(3.65, ax.get_ylim()[1] * 0.93, "threshold (score > 3)", fontsize=8, va="top")
    ax.set_xlabel("Q-CHAT-10 score")
    ax.set_ylabel("Toddlers")
    ax.set_xticks(range(11))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_item_rates(X, y, path: str) -> None:
    rates = pd.DataFrame({"ASD traits": X.loc[y == 1, ITEM_COLS].mean(),
                          "No ASD traits": X.loc[y == 0, ITEM_COLS].mean()})
    fig, ax = plt.subplots(figsize=(6, 3.2))
    rates.plot.bar(ax=ax, color=["#c0392b", "#2c7fb8"], width=0.75)
    ax.set_ylabel("Proportion of trait answers")
    ax.set_xlabel("Q-CHAT-10 item")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history_band(histories: list[dict], path: str) -> None:
    """Mean +- SD learning curves across CNN initialisations (runs stop at different epochs)."""
    n = max(len(h["loss"]) for h in histories)

    def stack(key):
        a = np.full((len(histories), n), np.nan)
        for i, h in enumerate(histories):
            a[i, :len(h[key])] = h[key]
        return np.nanmean(a, 0), np.nanstd(a, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, n + 1)
    for ax, key, title in [(axes[0], "loss", "CNN loss"), (axes[1], "accuracy", "CNN accuracy")]:
        for split, colour in [("", "#2c7fb8"), ("val_", "#e67e22")]:
            m, s = stack(split + key)
            ax.plot(ep, m, color=colour, label="validation" if split else "train")
            ax.fill_between(ep, m - s, m + s, color=colour, alpha=0.2, lw=0)
        ax.set_title(f"{title} (mean $\\pm$ SD, {len(histories)} initialisations)")
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, coef: pd.Series, path: str) -> None:
    """(a) permutation importance mean +- SD, (b) logistic-regression coefficients."""
    order = imp.xs("mean", axis=1, level=1)["1D CNN"].sort_values().index
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    yy = np.arange(len(order))
    for k, (model, colour) in enumerate([("Logistic Regression", "#27ae60"), ("Random Forest", "#7f8c8d"),
                                         ("1D CNN", "#c0392b")]):
        a.barh(yy + (k - 1) * 0.27, imp.loc[order, (model, "mean")], 0.27, xerr=imp.loc[order, (model, "std")],
               color=colour, label=SHORT[model], error_kw={"lw": 0.7, "capsize": 1.5})
    a.set_yticks(yy, order)
    a.set_xlabel("Decrease in ROC-AUC when permuted")
    a.set_title("(a) Permutation importance", fontsize=10)
    a.legend(frameon=False, loc="lower right", fontsize=8)
    c = coef.sort_values()
    b.barh(np.arange(len(c)), c.values, color=["#c0392b" if v > 0 else "#2c7fb8" for v in c.values])
    b.set_yticks(np.arange(len(c)), c.index)
    b.axvline(0, color="k", lw=0.7)
    b.set_xlabel("Coefficient (log-odds)")
    b.set_title("(b) Logistic-regression coefficients", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def table_main(e2: pd.DataFrame, nsplits: int) -> tuple[str, pd.DataFrame]:
    keys = ["accuracy", "recall (sensitivity)", "specificity", "f1", "roc_auc"]
    g = e2.groupby("model")
    agg = g[keys].agg(["mean", "std"])
    order = agg[("f1", "mean")].sort_values(ascending=False).index
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Test-set performance over {nsplits} repeated stratified 80/20 splits of the synthetic "
             r"dataset: mean $\pm$ SD, and the 2.5th--97.5th percentile range of accuracy across splits. "
             r"Rows are sorted by F1-score.}", r"\label{tab:main}", r"\small",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & Accuracy & Acc.\ range & Sensitivity & Specificity & F1 & AUC \\", r"\midrule"]
    for m in order:
        acc = e2.loc[e2.model == m, "accuracy"]
        cells = [f"{agg.loc[m, (k, 'mean')]:.3f} $\\pm$ {agg.loc[m, (k, 'std')]:.3f}" for k in keys]
        cells.insert(1, f"{acc.quantile(0.025):.3f}--{acc.quantile(0.975):.3f}")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines), agg


def table_clinical(y_te, probas: dict) -> tuple[str, pd.DataFrame]:
    ref = (probas["Logistic Regression"] >= 0.5).astype(int) == y_te
    rows = []
    for m, pr in probas.items():
        pred = (pr >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_te == 1)).sum())
        fn = int(((pred == 0) & (y_te == 1)).sum())
        tn = int(((pred == 0) & (y_te == 0)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        s_lo, s_hi = wilson(tp, tp + fn)
        c_lo, c_hi = wilson(tn, tn + fp)
        correct = pred == y_te
        b, c = int((ref & ~correct).sum()), int((~ref & correct).sum())
        p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
        rows.append({
            "model": m, "tp": tp, "fn": fn, "tn": tn, "fp": fp, "sens": sens, "sens_lo": s_lo, "sens_hi": s_hi,
            "spec": spec, "spec_lo": c_lo, "spec_hi": c_hi,
            "ppv": tp / (tp + fp) if tp + fp else float("nan"), "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "lr_pos": sens / (1 - spec) if spec < 1 else float("inf"),
            "lr_neg": (1 - sens) / spec if spec > 0 else float("inf"),
            "ppv_pop": ppv_at(sens, spec, POP_PREVALENCE), "ppv_pop_lo": ppv_at(s_lo, c_lo, POP_PREVALENCE),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        })
    df = pd.DataFrame(rows).set_index("model")
    n_pos, n_neg = int(y_te.sum()), int((1 - y_te).sum())
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Screening metrics on the E1 test split ($n={len(y_te)}$: {n_pos} positive, {n_neg} "
             r"negative) with Wilson 95\% confidence intervals. PPV and NPV are at the dataset's prevalence; "
             rf"PPV$_{{3\%}}$ re-weights the same sensitivity and specificity to a {POP_PREVALENCE * 100:.0f}\% "
             r"population prevalence (point estimate / value at the lower CI bounds). $p$: exact McNemar test "
             r"against logistic regression.}",
             r"\label{tab:clinical}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Model & Sensitivity (95\% CI) & Specificity (95\% CI) & PPV & NPV & LR+ & LR$-$ & "
             r"PPV$_{3\%}$ & $p$ \\", r"\midrule"]
    for m, r in df.sort_values("mcnemar_p", ascending=False).iterrows():
        lrp = r"$\infty$" if np.isinf(r.lr_pos) else f"{r.lr_pos:.1f}"
        p = "--" if m == "Logistic Regression" else ("1.000" if r.mcnemar_p >= 0.9995 else f"{r.mcnemar_p:.3f}"
                                                     if r.mcnemar_p >= 0.001 else "$<$0.001")
        lines.append(f"{m} & {f3(r.sens)} {ci(r.sens_lo, r.sens_hi)} & {f3(r.spec)} {ci(r.spec_lo, r.spec_hi)} & "
                     f"{f3(r.ppv)} & {f3(r.npv)} & {lrp} & {r.lr_neg:.3f} & {r.ppv_pop:.3f} / {r.ppv_pop_lo:.3f} & "
                     f"{p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines), df


def table_ablation(e3: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    agg3 = e3.groupby(["subset", "model"])[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"])
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Feature-subset ablation (mean $\pm$ SD over {e3['seed'].nunique()} splits).}}",
             r"\label{tab:ablation}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccc}", r"\toprule", r"Model & Accuracy & F1 & AUC \\", r"\midrule"]
    for i, sname in enumerate(SUBSETS):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{sname}}}}} \\")
        for m in ["Logistic Regression", "Random Forest", "1D CNN"]:
            r = agg3.loc[(sname, m)]
            cells = [f"{r[(k, 'mean')]:.3f} $\\pm$ {r[(k, 'std')]:.3f}" for k in ("accuracy", "f1", "roc_auc")]
            lines.append(r"\quad " + m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), agg3


def table_params(n_features: int) -> tuple[str, int]:
    model = build_cnn(n_features)
    kinds = {"Conv1D": "Conv1D", "MaxPooling1D": "MaxPool1D", "Dropout": "Dropout", "Flatten": "Flatten",
             "Dense": "Dense"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Layer-by-layer structure of the 1D CNN, taken from the executable model "
             r"(\texttt{model.summary()}).}", r"\label{tab:params}", r"\footnotesize",
             r"\begin{tabular}{llr}", r"\toprule", r"Layer (configuration) & Output shape & Parameters \\",
             r"\midrule", rf"Input & ${n_features}\times 1$ & 0 \\"]
    for layer in model.layers:
        cls = type(layer).__name__
        cfg = layer.get_config()
        if cls == "Conv1D":
            desc = f"Conv1D ({cfg['filters']}, $k$={cfg['kernel_size'][0]}, ReLU)"
        elif cls == "Dense":
            desc = f"Dense ({cfg['units']}, {cfg['activation']})"
        elif cls == "Dropout":
            desc = f"Dropout ({cfg['rate']})"
        elif cls == "MaxPooling1D":
            desc = f"MaxPool1D (size {cfg['pool_size'][0]})"
        else:
            desc = kinds.get(cls, cls)
        shape = r"$\times$".join(str(d) for d in layer.output.shape[1:])
        lines.append(f"{desc} & {shape} & {layer.count_params():,} \\\\".replace(",", "{,}"))
    total = int(model.count_params())
    lines += [r"\midrule", rf"Total (all trainable) & & {total:,} \\".replace(",", "{,}"),
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), total


def environment() -> dict:
    import tensorflow as tf

    cpu = platform.processor() or "unknown CPU"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"python": platform.python_version(), "tensorflow": tf.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "cpu": cpu, "ncpu": os.cpu_count(),
            "os": f"{platform.system()} {platform.release().split('-')[0]}"}


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--splits", type=int, default=20, help="Repeated stratified splits (seeds 0..n-1)")
    p.add_argument("--importance-splits", type=int, default=5)
    p.add_argument("--permutations", type=int, default=30)
    p.add_argument("--cnn-inits", type=int, default=10)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--paper-dir", default="paper")
    args = p.parse_args()

    gen = os.path.join(args.paper_dir, "generated")
    figs = os.path.join(args.paper_dir, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    seeds = list(range(args.splits))

    df = make_synthetic(seed=42) if args.synthetic else load_csv(args.data)
    X, y = clean(df)
    plot_item_rates(X, y, os.path.join(figs, "item_rates.pdf"))
    plot_score_hist(X, y, os.path.join(figs, "score_hist.pdf"))

    # E1 ---------------------------------------------------------------
    X_tr, X_te, y_tr, y_te, pre, _ = split_encode(X, y, ALL_COLS, 42, args.test_size)
    n_feat = X_tr.shape[1]
    probas, fitted, _ = fit_models(X_tr, X_te, y_tr, 42, args.epochs)
    plot_confusion_matrices(probas, y_te, os.path.join(figs, "confusion_matrices.pdf"))
    plot_roc(probas, y_te, os.path.join(figs, "roc_curves.pdf"), zoom=True)
    tex, clin = table_clinical(y_te, probas)
    write(os.path.join(gen, "table_clinical.tex"), tex)
    clin.to_csv(os.path.join(gen, "e1_clinical.csv"))
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    lr_coef = pd.Series(fitted["Logistic Regression"].coef_[0], index=names)
    print("E1 done", flush=True)

    # E2 + E4 ----------------------------------------------------------
    rows, imp_raw, test_ids = [], {m: {c: [] for c in ALL_COLS} for m in
                                   ["Logistic Regression", "Random Forest", "1D CNN"]}, set()
    rng = np.random.default_rng(0)
    for seed in seeds:
        a, b, c, d, pre_s, X_te_raw = split_encode(X, y, ALL_COLS, seed, args.test_size)
        test_ids.update(X_te_raw.index)
        pr, fit, _ = fit_models(a, b, c, seed, args.epochs)
        rows += [{"model": m, "seed": seed, **metrics(d, v)} for m, v in pr.items()]
        if seed < args.importance_splits:
            for m in imp_raw:
                for col, drops in permutation_importance(predictor(fit[m]), pre_s, X_te_raw, d,
                                                         args.permutations, rng).items():
                    imp_raw[m][col] += drops
        print(f"E2 split {seed} done", flush=True)
    e2 = pd.DataFrame(rows)
    e2.to_csv(os.path.join(gen, "e2_repeated.csv"), index=False)
    tex, agg = table_main(e2, len(seeds))
    write(os.path.join(gen, "table_main.tex"), tex)
    imp = pd.concat({m: pd.DataFrame({c: [np.mean(v), np.std(v)] for c, v in cols.items()},
                                     index=["mean", "std"]).T for m, cols in imp_raw.items()}, axis=1)
    imp = imp.rename(index=PRETTY)
    imp.to_csv(os.path.join(gen, "e4_importance.csv"))
    coef_show = lr_coef[ITEM_COLS + BINARY_COLS + NUMERIC_COLS].rename(index=PRETTY)
    onehot_max = lr_coef.drop(ITEM_COLS + BINARY_COLS + NUMERIC_COLS).abs().max()
    plot_importance(imp, coef_show, os.path.join(figs, "importance.pdf"))

    # E3 ---------------------------------------------------------------
    rows = []
    for sname, cols in SUBSETS.items():
        for seed in seeds:
            a, b, c, d, _, _ = split_encode(X, y, cols, seed, args.test_size)
            pr, _, _ = fit_models(a, b, c, seed, args.epochs, only={"Logistic Regression", "Random Forest", "1D CNN"})
            rows += [{"subset": sname, "model": m, "seed": seed, **metrics(d, v)} for m, v in pr.items()]
        print(f"E3 {sname} done", flush=True)
    e3 = pd.DataFrame(rows)
    e3.to_csv(os.path.join(gen, "e3_ablation.csv"), index=False)
    tex, agg3 = table_ablation(e3)
    write(os.path.join(gen, "table_ablation.tex"), tex)

    # E5 ---------------------------------------------------------------
    hists, e5 = [], []
    for init in range(args.cnn_inits):
        cnn, h = train_cnn(X_tr, y_tr, epochs=args.epochs, seed=1000 + init)
        hists.append(h)
        e5.append({"init": init, "epochs": len(h["loss"]), **metrics(y_te, cnn_predict_proba(cnn, X_te))})
    e5 = pd.DataFrame(e5)
    e5.to_csv(os.path.join(gen, "e5_cnn_inits.csv"), index=False)
    plot_history_band(hists, os.path.join(figs, "cnn_training.pdf"))

    # Parameters, environment, macros --------------------------------------
    tex, n_params = table_params(n_feat)
    write(os.path.join(gen, "table_params.tex"), tex)
    env = environment()
    item_imp = imp.loc[ITEM_COLS]
    demo_imp = imp.drop(item_imp.index)
    cnn_row = agg.loc["1D CNN"]
    order = agg[("f1", "mean")].sort_values(ascending=False).index
    lr_clin = clin.loc["Logistic Regression"]
    cnn_clin = clin.loc["1D CNN"]
    ab = lambda s, m, k: f"{agg3.loc[(s, m), (k, 'mean')]:.3f}"  # noqa: E731
    macros = {
        "DataSource": "synthetic" if args.synthetic else "real",
        "NRecords": len(df), "NASD": int(y.sum()), "NNoASD": int((1 - y).sum()),
        "PctASD": f"{100 * y.mean():.1f}", "NFeat": n_feat,
        "MalePct": f"{(X['Sex'] == 1).mean() * 100:.1f}", "AgeMean": f"{X['Age_Mons'].mean():.1f}",
        "AgeStd": f"{X['Age_Mons'].std():.1f}", "JaundicePct": f"{X['Jaundice'].mean() * 100:.1f}",
        "FamilyPct": f"{X['Family_mem_with_ASD'].mean() * 100:.1f}",
        "NSplits": len(seeds), "NTest": len(y_te), "NTestPos": int(y_te.sum()), "NTestNeg": int((1 - y_te).sum()),
        "NUniqueTest": len(test_ids), "NImpSplits": args.importance_splits, "NPerm": args.permutations,
        "NInits": args.cnn_inits,
        "CNNParams": f"{n_params:,}".replace(",", "{,}"),
        "CNNAcc": f3(cnn_row[("accuracy", "mean")]), "CNNAccStd": f3(cnn_row[("accuracy", "std")]),
        "CNNFone": f3(cnn_row[("f1", "mean")]), "CNNAUC": f3(cnn_row[("roc_auc", "mean")]),
        "CNNSens": f3(cnn_row[("recall (sensitivity)", "mean")]), "CNNSpec": f3(cnn_row[("specificity", "mean")]),
        "LRAcc": f3(agg.loc["Logistic Regression", ("accuracy", "mean")]),
        "LRAccStd": f3(agg.loc["Logistic Regression", ("accuracy", "std")]),
        "BestModel": order[0], "WorstModel": order[-1],
        "WorstAcc": f3(agg.loc[order[-1], ("accuracy", "mean")]),
        "LRSpecLo": f3(lr_clin.spec_lo), "LRSensLo": f3(lr_clin.sens_lo),
        "LRPPVpopLo": f3(lr_clin.ppv_pop_lo), "CNNPPVpopLo": f3(cnn_clin.ppv_pop_lo),
        "CNNMcNemarP": "1.000" if cnn_clin.mcnemar_p >= 0.9995 else f"{cnn_clin.mcnemar_p:.3f}",
        "DemoCNNAUC": ab("Demographics only", "1D CNN", "roc_auc"),
        "DemoRFAUC": ab("Demographics only", "Random Forest", "roc_auc"),
        "DemoLRAUC": ab("Demographics only", "Logistic Regression", "roc_auc"),
        "DemoCNNAcc": ab("Demographics only", "1D CNN", "accuracy"),
        "ItemsCNNAcc": ab("Q-CHAT-10 items only", "1D CNN", "accuracy"),
        "ItemsLRAcc": ab("Q-CHAT-10 items only", "Logistic Regression", "accuracy"),
        "TopItemCNN": item_imp[("1D CNN", "mean")].idxmax(),
        "MinItemImpCNN": f3(item_imp[("1D CNN", "mean")].min()),
        "MaxItemImpCNN": f3(item_imp[("1D CNN", "mean")].max()),
        "MaxDemoImp": (lambda v: "$<$0.001" if v < 0.0005 else f3(v))(demo_imp.xs("mean", axis=1, level=1).abs().max().max()),
        "ItemCoefMin": f"{lr_coef[ITEM_COLS].min():.2f}", "ItemCoefMax": f"{lr_coef[ITEM_COLS].max():.2f}",
        "DemoCoefMax": f"{max(lr_coef[BINARY_COLS + NUMERIC_COLS].abs().max(), onehot_max):.2f}",
        "InitAccMean": f3(e5.accuracy.mean()), "InitAccStd": f3(e5.accuracy.std()),
        "InitAccMin": f3(e5.accuracy.min()), "InitEpochMin": int(e5.epochs.min()), "InitEpochMax": int(e5.epochs.max()),
        "EnvPython": env["python"], "EnvTF": env["tensorflow"], "EnvSklearn": env["sklearn"],
        "EnvNumpy": env["numpy"], "EnvPandas": env["pandas"], "EnvCPU": env["cpu"].replace("(R)", "").replace("(TM)", ""),
        "EnvNCPU": env["ncpu"], "EnvOS": env["os"],
    }
    write(os.path.join(gen, "macros.tex"), "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    with open(os.path.join(gen, "run_info.json"), "w") as f:
        json.dump({"args": vars(args), "environment": env, "macros": {k: str(v) for k, v in macros.items()}}, f,
                  indent=2)
    print(json.dumps({k: str(v) for k, v in macros.items()}, indent=1))


if __name__ == "__main__":
    main()
