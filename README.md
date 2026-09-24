# Autism Spectrum Disorder Classification for Toddlers (CNN + Machine Learning)

A screening model that predicts whether a toddler (12–36 months) shows autism spectrum disorder (ASD) traits, using the
**Q-CHAT-10** questionnaire plus demographic data. It trains a **1D convolutional neural network** and seven classical
machine-learning models on the same data and compares them.

> **Disclaimer:** this is a research and education project. It is a screening aid, not a diagnostic tool. A positive
> result means "refer for clinical assessment."

## Dataset

The code expects the public *Autism screening data for toddlers* dataset (Thabtah, 2018; 1054 records), usually named
`Toddler Autism dataset July 2018.csv` (available on Kaggle). Put it in `data/`.

| Column | Meaning |
|---|---|
| `A1`–`A10` | Q-CHAT-10 items, coded 1 = answer indicates an ASD trait |
| `Age_Mons` | Age in months |
| `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`, `Who completed the test` | Demographics and history |
| `Qchat-10-Score` | Sum of A1–A10 (**dropped**: it defines the label) |
| `Class/ASD Traits` | Target: `Yes` if the score is > 3 |

If you don't have the CSV, `--synthetic` builds a dataset with the same schema and labelling rule, so the pipeline runs
end to end. Numbers from synthetic data are not research results.

## Method

1. **Preprocessing** (`asd/data.py`): normalise column names; binary-encode sex, jaundice and family history; one-hot
   encode ethnicity and respondent; min-max scale age; drop `Case_No` and the leaky `Qchat-10-Score`. Use a stratified
   80/20 train/test split, with the encoders fit on the training set only.
2. **Classical ML** (`asd/models.py`): Logistic Regression, SVM (RBF, Platt-calibrated), KNN, Decision Tree, Random Forest,
   Gradient Boosting and Gaussian Naive Bayes, each scored with 5-fold stratified cross-validation on the training set.
3. **1D CNN** (`asd/models.py`): the encoded feature vector is treated as a 1-channel sequence.

   ```
   Input(29×1) → Conv1D(32,3) → Conv1D(64,3) → MaxPool(2) → Dropout(0.25)
               → Conv1D(64,3) → Flatten → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)
   ```

   It uses the Adam optimiser and binary cross-entropy loss, holds out 15% of the training set for validation, and applies
   early stopping (patience 20, best weights restored) and ReduceLROnPlateau.
4. **Evaluation** (`asd/evaluate.py`): accuracy, precision, recall (sensitivity), specificity, F1 and ROC-AUC on the held-out
   test set, plus confusion matrices, ROC curves and CNN learning curves.

## Usage

```bash
pip install -r requirements.txt

# Train and compare every model
python train.py --data "data/Toddler Autism dataset July 2018.csv"
python train.py --synthetic            # without the real dataset

# Screen one toddler with a saved model (cnn, logistic_regression, random_forest, svm_rbf, ...)
python predict.py --answers 1 1 0 1 1 0 1 1 0 1 --age 24 --sex m \
    --jaundice no --family-asd no --ethnicity "White European" --model cnn

pytest -q
```

`train.py` writes to `outputs/`: `metrics.csv` / `metrics.json`, `confusion_matrices.png`, `roc_curves.png`,
`cnn_training.png`, the trained models (`*.joblib`, `cnn.keras`) and the fitted `preprocessor.joblib`.

## Example results (synthetic data, seed 42, 211 test records)

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVM (RBF) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **1D CNN** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 0.976 | 1.000 | 0.924 | 0.983 | 1.000 |
| Random Forest | 0.967 | 1.000 | 0.894 | 0.976 | 1.000 |
| KNN | 0.962 | 0.979 | 0.924 | 0.973 | 0.995 |
| Naive Bayes | 0.919 | 0.986 | 0.773 | 0.944 | 0.942 |
| Decision Tree | 0.867 | 0.924 | 0.742 | 0.905 | 0.871 |

**Why the scores are near-perfect:** in this dataset the label is a deterministic rule (`sum(A1..A10) > 3`). Any model
that can learn a linear threshold over the item answers (logistic regression, SVM, the CNN) will reach about 100%. Tree
models and Naive Bayes do worse because they approximate that sum with axis-aligned splits or an independence assumption.
Expect the same pattern on the real CSV. For a harder, more clinically meaningful benchmark, you could predict clinical
diagnosis instead of the questionnaire-derived label, or train without some of the items.

## Research paper

`paper/` contains a manuscript in Elsevier double-column format (`elsarticle`, `5p`): `main.tex` and the compiled
`main.pdf`. It is framed as a methodological study on a synthetic surrogate: **every number in it comes from
synthetic data**, and it says so throughout. `paper/response_to_reviewers.md` answers the first round of review.

`experiments.py` produces every number, table and figure:

| Experiment | What it does |
|---|---|
| E1 | One held-out split: confusion matrices, ROC curves with a zoomed inset, Wilson 95% CIs, PPV/NPV/LR+/LR-, PPV at 3% prevalence, exact McNemar tests against logistic regression |
| E2 | All 8 models over 20 repeated stratified splits (mean, SD, percentile range) |
| E3 | Feature-subset ablation (items only / demographics only / all) over the same splits |
| E4 | Permutation importance (LR, RF, CNN; 5 splits x 30 permutations), plus logistic-regression coefficients |
| E5 | CNN retrained from 10 initialisations (learning-curve bands) |

```bash
python experiments.py --synthetic                      # ~45 min on a 4-core CPU
cd paper && pdflatex main.tex && pdflatex main.tex
```

Running `experiments.py --data "data/Toddler Autism dataset July 2018.csv"` regenerates all numbers on the public
data, but the paper's text then needs rewriting: it describes the synthetic surrogate throughout.

## Project layout

```
asd/data.py        loading, synthetic data, cleaning, encoding, split
asd/models.py      classical ML models and the 1D CNN
asd/evaluate.py    metrics and plots
train.py           training and comparison CLI
predict.py         single-toddler screening CLI
experiments.py     experiments, tables and figures for the paper
paper/             Elsevier LaTeX manuscript
tests/             pytest suite
```

## Reference

Thabtah, F. (2018). *Autism screening data for toddlers*. Q-CHAT-10 based; see also Allison et al. (2012), "Toward brief
'Red Flags' for autism screening: The Short Autism Spectrum Quotient and the Short Quantitative Checklist in 1,000 cases
and 3,000 controls", *JAACAP* 51(2).
