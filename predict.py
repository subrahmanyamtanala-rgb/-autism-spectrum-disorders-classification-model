"""Screen a single toddler with a trained model.

Example:
    python predict.py --answers 1 1 0 1 1 0 1 1 0 1 --age 24 --sex m \
        --jaundice no --family-asd no --ethnicity "White European" --model cnn
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from asd.data import ITEM_COLS
from asd.models import cnn_predict_proba


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answers", type=int, nargs=10, required=True, metavar="A",
                   help="Q-CHAT-10 items A1..A10 coded 1 = ASD trait present, 0 = absent")
    p.add_argument("--age", type=int, required=True, help="Age in months")
    p.add_argument("--sex", choices=["m", "f"], required=True)
    p.add_argument("--jaundice", choices=["yes", "no"], default="no")
    p.add_argument("--family-asd", choices=["yes", "no"], default="no")
    p.add_argument("--ethnicity", default="Others")
    p.add_argument("--respondent", default="family member")
    p.add_argument("--model", default="cnn", help="'cnn' or a saved .joblib model name, e.g. random_forest")
    p.add_argument("--models-dir", default="outputs")
    args = p.parse_args()

    if any(a not in (0, 1) for a in args.answers):
        p.error("--answers must be ten values of 0 or 1")

    bundle = joblib.load(os.path.join(args.models_dir, "preprocessor.joblib"))
    row = dict(zip(ITEM_COLS, args.answers))
    row.update({
        "Age_Mons": args.age, "Sex": int(args.sex == "m"), "Jaundice": int(args.jaundice == "yes"),
        "Family_mem_with_ASD": int(args.family_asd == "yes"),
        "Ethnicity": args.ethnicity.strip().lower(), "Who completed the test": args.respondent.strip().lower(),
    })
    X = bundle["preprocessor"].transform(pd.DataFrame([row])[bundle["input_columns"]]).astype("float32")

    if args.model == "cnn":
        import tensorflow as tf

        model = tf.keras.models.load_model(os.path.join(args.models_dir, "cnn.keras"))
        proba = float(cnn_predict_proba(model, X)[0])
    else:
        model = joblib.load(os.path.join(args.models_dir, f"{args.model}.joblib"))
        proba = float(model.predict_proba(X)[0, 1])

    label = "ASD traits likely - refer for clinical assessment" if proba >= 0.5 else "ASD traits unlikely"
    print(f"Q-CHAT-10 score: {int(np.sum(args.answers))}/10")
    print(f"Model ({args.model}) probability of ASD traits: {proba:.3f}")
    print(f"Screening result: {label}")
    print("Note: this is a screening aid, not a diagnosis.")


if __name__ == "__main__":
    main()
