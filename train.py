"""Train and compare the CNN and classical ML models for toddler ASD screening.

Usage:
    python train.py --data "data/Toddler Autism dataset July 2018.csv"
    python train.py --synthetic          # run without the real dataset
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from asd.data import load_csv, make_synthetic, prepare
from asd.evaluate import metrics, plot_confusion_matrices, plot_history, plot_roc
from asd.models import cnn_predict_proba, ml_models, train_cnn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Path to the Q-CHAT-10 toddler CSV")
    src.add_argument("--synthetic", action="store_true", help="Use a generated dataset with the same schema")
    p.add_argument("--out", default="outputs", help="Directory for models, metrics and plots")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv", type=int, default=5, help="Folds for cross-validating the classical models (0 = skip)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = make_synthetic(seed=args.seed) if args.synthetic else load_csv(args.data)
    print(f"Loaded {len(df)} records ({'synthetic' if args.synthetic else args.data})")
    ds = prepare(df, test_size=args.test_size, seed=args.seed)
    print(f"Train {ds.X_train.shape}, test {ds.X_test.shape}, ASD prevalence {ds.y_train.mean():.2%}")

    probas, rows = {}, []

    for name, model in ml_models(args.seed).items():
        cv = {}
        if args.cv:
            folds = StratifiedKFold(args.cv, shuffle=True, random_state=args.seed)
            scores = cross_val_score(model, ds.X_train, ds.y_train, cv=folds, scoring="accuracy")
            cv = {"cv_accuracy_mean": scores.mean(), "cv_accuracy_std": scores.std()}
        model.fit(ds.X_train, ds.y_train)
        probas[name] = model.predict_proba(ds.X_test)[:, 1]
        rows.append({"model": name, **metrics(ds.y_test, probas[name]), **cv})
        joblib.dump(model, os.path.join(args.out, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"))
        print(f"  {name:<20} acc={rows[-1]['accuracy']:.4f}")

    cnn, history = train_cnn(ds.X_train, ds.y_train, epochs=args.epochs, seed=args.seed)
    probas["1D CNN"] = cnn_predict_proba(cnn, ds.X_test)
    rows.append({"model": "1D CNN", **metrics(ds.y_test, probas["1D CNN"]), "epochs_trained": len(history["loss"])})
    cnn.save(os.path.join(args.out, "cnn.keras"))
    print(f"  {'1D CNN':<20} acc={rows[-1]['accuracy']:.4f} ({len(history['loss'])} epochs)")

    joblib.dump({"preprocessor": ds.preprocessor, "feature_names": ds.feature_names,
                 "input_columns": ds.input_columns}, os.path.join(args.out, "preprocessor.joblib"))

    table = pd.DataFrame(rows).set_index("model").sort_values(["f1", "roc_auc"], ascending=False)
    table.to_csv(os.path.join(args.out, "metrics.csv"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(json.loads(table.to_json(orient="index")), f, indent=2)
    plot_confusion_matrices(probas, ds.y_test, os.path.join(args.out, "confusion_matrices.png"))
    plot_roc(probas, ds.y_test, os.path.join(args.out, "roc_curves.png"))
    plot_history(history, os.path.join(args.out, "cnn_training.png"))

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200, "display.max_columns", 20):
        print("\nTest-set results\n" + str(table))
    print(f"\nBest model by F1: {table.index[0]}. Artefacts written to {args.out}/")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
