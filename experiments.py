"""Run the experiments reported in the paper and write its tables and figures.

    python experiments.py --synthetic
    python experiments.py --data "data/Toddler Autism dataset July 2018.csv"

Outputs go to paper/generated/ (LaTeX tables, CSVs) and paper/figures/.
Experiments:
  E1  all models, seed 42 split: confusion matrices, ROC, CNN learning curves
  E2  all models over repeated stratified splits (mean +- std)
  E3  feature-subset ablation (Q-CHAT items / demographics / all)
  E4  permutation importance of the input columns (Random Forest and CNN)
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from asd.data import (
    BINARY_COLS, CATEGORICAL_COLS, ITEM_COLS, NUMERIC_COLS, build_preprocessor, clean, load_csv,
    make_synthetic, prepare,
)
from asd.evaluate import metrics, plot_confusion_matrices, plot_history, plot_roc
from asd.models import cnn_predict_proba, ml_models, train_cnn

SUBSETS = {
    "Q-CHAT-10 items only": ITEM_COLS,
    "Demographics only": NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS,
    "All features": ITEM_COLS + NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS,
}
METRIC_KEYS = ["accuracy", "precision", "recall (sensitivity)", "specificity", "f1", "roc_auc"]
METRIC_HDR = ["Acc.", "Prec.", "Sens.", "Spec.", "F1", "AUC"]


def subset_preprocessor(cols):
    parts = []
    if any(c in ITEM_COLS + BINARY_COLS for c in cols):
        parts.append(("passthrough", "passthrough", [c for c in cols if c in ITEM_COLS + BINARY_COLS]))
    full = build_preprocessor()
    for name, tr, tcols in full.transformers:
        if name != "passthrough" and any(c in cols for c in tcols):
            parts.append((name, clone(tr), [c for c in tcols if c in cols]))
    return ColumnTransformer(parts)


def run_all_models(X_tr, X_te, y_tr, y_te, seed, epochs, only=None):
    probas, hist = {}, None
    for name, model in ml_models(seed).items():
        if only and name not in only:
            continue
        probas[name] = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    if not only or "1D CNN" in only:
        cnn, hist = train_cnn(X_tr, y_tr, epochs=epochs, seed=seed)
        probas["1D CNN"] = cnn_predict_proba(cnn, X_te)
        hist["_model"] = cnn
    return probas, hist


def split_encode(X, y, cols, seed, test_size):
    X_tr, X_te, y_tr, y_te = train_test_split(X[cols], y, test_size=test_size, stratify=y, random_state=seed)
    pre = subset_preprocessor(cols)
    return pre.fit_transform(X_tr).astype("float32"), pre.transform(X_te).astype("float32"), y_tr, y_te, pre, X_te


def fmt_table(df: pd.DataFrame, caption: str, label: str, bold_best=True) -> str:
    """Render a DataFrame of 'mean' or 'mean ± std' strings as a booktabs table."""
    cols = "l" + "c" * len(df.columns)
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join([df.index.name or ""] + list(df.columns)) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(" & ".join([str(idx)] + [str(v) for v in row]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def write_ablation_table(e3: pd.DataFrame, path: str) -> None:
    agg3 = e3.groupby(["subset", "model"])[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"])
    nseeds = e3["seed"].nunique()
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Feature-subset ablation (mean $\pm$ std over {nseeds} splits).}}", r"\label{tab:ablation}",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & Accuracy & F1 & AUC \\", r"\midrule"]
    for i, sname in enumerate(SUBSETS):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{sname}}}}} \\")
        for m in ["Logistic Regression", "Random Forest", "1D CNN"]:
            r = agg3.loc[(sname, m)]
            cells = [f"{r[(k, 'mean')]:.3f} $\\pm$ {r[(k, 'std')]:.3f}" for k in ("accuracy", "f1", "roc_auc")]
            lines.append(r"\quad " + m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_score_hist(X, y, path: str) -> None:
    score = X[ITEM_COLS].sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.0))
    bins = np.arange(-0.5, 11.5, 1)
    ax.hist([score[y == 0], score[y == 1]], bins=bins, stacked=True, color=["#2c7fb8", "#c0392b"],
            label=["No ASD", "ASD"])
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
    item_rates = pd.DataFrame({
        "ASD": X.loc[y == 1, ITEM_COLS].mean(), "No ASD": X.loc[y == 0, ITEM_COLS].mean()})
    fig, ax = plt.subplots(figsize=(6, 3.2))
    item_rates.plot.bar(ax=ax, color=["#c0392b", "#2c7fb8"], width=0.75)
    ax.set_ylabel("Proportion of trait answers")
    ax.set_xlabel("Q-CHAT-10 item")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--paper-dir", default="paper")
    args = p.parse_args()

    gen = os.path.join(args.paper_dir, "generated")
    figs = os.path.join(args.paper_dir, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)

    df = make_synthetic(seed=42) if args.synthetic else load_csv(args.data)
    X, y = clean(df)

    # Dataset description
    stats = {
        "source": "synthetic" if args.synthetic else os.path.basename(args.data),
        "n": int(len(df)), "n_asd": int(y.sum()), "n_no": int((1 - y).sum()),
        "n_features_encoded": int(prepare(df, seed=42).X_train.shape[1]),
        "male_pct": float((X["Sex"] == 1).mean() * 100),
        "age_mean": float(X["Age_Mons"].mean()), "age_std": float(X["Age_Mons"].std()),
        "jaundice_pct": float(X["Jaundice"].mean() * 100),
        "family_pct": float(X["Family_mem_with_ASD"].mean() * 100),
    }
    plot_item_rates(X, y, os.path.join(figs, "item_rates.pdf"))
    plot_score_hist(X, y, os.path.join(figs, "score_hist.pdf"))

    # E1: single split, figures
    all_cols = SUBSETS["All features"]
    X_tr, X_te, y_tr, y_te, pre, _ = split_encode(X, y, all_cols, 42, args.test_size)
    probas, hist = run_all_models(X_tr, X_te, y_tr, y_te, 42, args.epochs)
    plot_confusion_matrices(probas, y_te, os.path.join(figs, "confusion_matrices.pdf"))
    plot_roc(probas, y_te, os.path.join(figs, "roc_curves.pdf"))
    plot_history(hist, os.path.join(figs, "cnn_training.pdf"))
    e1 = pd.DataFrame({m: metrics(y_te, pr) for m, pr in probas.items()}).T[METRIC_KEYS]
    e1.to_csv(os.path.join(gen, "e1_single_split.csv"))
    stats["cnn_epochs"] = len(hist["loss"])
    stats["cnn_params"] = int(hist["_model"].count_params())

    # E2: repeated splits
    rows = []
    for seed in args.seeds:
        a, b, c, d, _, _ = split_encode(X, y, all_cols, seed, args.test_size)
        pr, _ = run_all_models(a, b, c, d, seed, args.epochs)
        rows += [{"model": m, "seed": seed, **metrics(d, v)} for m, v in pr.items()]
    e2 = pd.DataFrame(rows)
    e2.to_csv(os.path.join(gen, "e2_repeated.csv"), index=False)
    agg = e2.groupby("model")[METRIC_KEYS].agg(["mean", "std"])
    order = agg[("f1", "mean")].sort_values(ascending=False).index
    tab = pd.DataFrame(index=pd.Index(order, name="Model"))
    for k, h in zip(METRIC_KEYS, METRIC_HDR):
        tab[h] = [f"{agg.loc[m, (k, 'mean')]:.3f} $\\pm$ {agg.loc[m, (k, 'std')]:.3f}" for m in order]
    with open(os.path.join(gen, "table_main.tex"), "w") as f:
        f.write(fmt_table(tab, f"Test-set performance of all classifiers over {len(args.seeds)} stratified "
                          "80/20 splits (mean $\\pm$ standard deviation). Sorted by F1-score.", "tab:main"))

    # E3: feature-subset ablation
    rows = []
    for sname, cols in SUBSETS.items():
        for seed in args.seeds:
            a, b, c, d, _, _ = split_encode(X, y, cols, seed, args.test_size)
            pr, _ = run_all_models(a, b, c, d, seed, args.epochs,
                                   only={"Logistic Regression", "Random Forest", "1D CNN"})
            rows += [{"subset": sname, "model": m, "seed": seed, **metrics(d, v)} for m, v in pr.items()]
    e3 = pd.DataFrame(rows)
    e3.to_csv(os.path.join(gen, "e3_ablation.csv"), index=False)
    agg3 = e3.groupby(["subset", "model"])[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"])
    write_ablation_table(e3, os.path.join(gen, "table_ablation.tex"))

    # E4: permutation importance on the seed-42 split (input columns, not encoded dummies)
    _, X_te_raw, _, y_te_raw = train_test_split(X[all_cols], y, test_size=args.test_size, stratify=y,
                                                random_state=42)
    rf = ml_models(42)["Random Forest"].fit(X_tr, y_tr)
    cnn = hist["_model"]
    predictors = {"Random Forest": lambda Z: rf.predict_proba(Z)[:, 1], "1D CNN": lambda Z: cnn_predict_proba(cnn, Z)}
    rng = np.random.default_rng(0)
    imp = {}
    for mname, fn in predictors.items():
        base = metrics(y_te_raw, fn(pre.transform(X_te_raw).astype("float32")))["roc_auc"]
        drops = {}
        for col in all_cols:
            vals = []
            for _ in range(10):
                Z = X_te_raw.copy()
                Z[col] = rng.permutation(Z[col].to_numpy())
                vals.append(base - metrics(y_te_raw, fn(pre.transform(Z).astype("float32")))["roc_auc"])
            drops[col] = np.mean(vals)
        imp[mname] = drops
    imp = pd.DataFrame(imp).rename(index={"Who completed the test": "Respondent", "Family_mem_with_ASD": "Family ASD",
                                          "Age_Mons": "Age"})
    imp.to_csv(os.path.join(gen, "e4_importance.csv"))
    imp = imp.sort_values("1D CNN")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    yy = np.arange(len(imp))
    ax.barh(yy - 0.2, imp["Random Forest"], 0.4, label="Random Forest", color="#7f8c8d")
    ax.barh(yy + 0.2, imp["1D CNN"], 0.4, label="1D CNN", color="#c0392b")
    ax.set_yticks(yy, imp.index)
    ax.set_xlabel("Mean decrease in ROC-AUC when permuted")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "importance.pdf"))
    plt.close(fig)

    with open(os.path.join(gen, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # LaTeX macros so the prose picks up the numbers.
    best = order[0]
    cnn_row = agg.loc["1D CNN"]
    macros = {
        "DataSource": "synthetic" if args.synthetic else "real",
        "NRecords": stats["n"], "NASD": stats["n_asd"], "NNoASD": stats["n_no"],
        "PctASD": f"{100 * stats['n_asd'] / stats['n']:.1f}", "NFeat": stats["n_features_encoded"],
        "MalePct": f"{stats['male_pct']:.1f}", "AgeMean": f"{stats['age_mean']:.1f}",
        "AgeStd": f"{stats['age_std']:.1f}", "JaundicePct": f"{stats['jaundice_pct']:.1f}",
        "FamilyPct": f"{stats['family_pct']:.1f}", "NSeeds": len(args.seeds),
        "CNNParams": f"{stats['cnn_params']:,}".replace(",", "{,}"), "CNNEpochs": stats["cnn_epochs"],
        "CNNAcc": f"{cnn_row[('accuracy', 'mean')]:.3f}", "CNNAccStd": f"{cnn_row[('accuracy', 'std')]:.3f}",
        "CNNFone": f"{cnn_row[('f1', 'mean')]:.3f}", "CNNAUC": f"{cnn_row[('roc_auc', 'mean')]:.3f}",
        "CNNSens": f"{cnn_row[('recall (sensitivity)', 'mean')]:.3f}",
        "CNNSpec": f"{cnn_row[('specificity', 'mean')]:.3f}",
        "BestModel": best, "WorstModel": order[-1],
        "WorstAcc": f"{agg.loc[order[-1], ('accuracy', 'mean')]:.3f}",
        "DemoCNNAUC": f"{agg3.loc[('Demographics only', '1D CNN'), ('roc_auc', 'mean')]:.3f}",
        "DemoRFAUC": f"{agg3.loc[('Demographics only', 'Random Forest'), ('roc_auc', 'mean')]:.3f}",
        "DemoLRAUC": f"{agg3.loc[('Demographics only', 'Logistic Regression'), ('roc_auc', 'mean')]:.3f}",
        "ItemsCNNAcc": f"{agg3.loc[('Q-CHAT-10 items only', '1D CNN'), ('accuracy', 'mean')]:.3f}",
        "TopItemCNN": imp["1D CNN"].idxmax(), "TopDemoCNN": imp.loc[~imp.index.str.startswith("A"), "1D CNN"].idxmax(),
    }
    with open(os.path.join(gen, "macros.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(tab.to_string())
    print(e3.groupby(["subset", "model"])[["accuracy", "f1", "roc_auc"]].mean().round(3).to_string())
    print(imp.round(4).to_string())
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
