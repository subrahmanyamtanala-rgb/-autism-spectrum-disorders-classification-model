# Response to the reviewer

**Manuscript:** *Reproducibility and label leakage in machine learning for Q-CHAT-10 toddler autism screening: a
synthetic-data study of a 1D CNN and classical classifiers*
(previous title: *Early screening of autism spectrum disorder in toddlers using a one-dimensional convolutional neural
network and machine learning on Q-CHAT-10 behavioural data*)

We thank the reviewer for a careful and constructive review. We agree with the central assessment: the manuscript's
contribution is its analysis of the Q-CHAT-10 benchmark, not the CNN. We have revised the paper around that
contribution. Each comment is answered below, and the manuscript location of each change is given. All numbers
quoted here come from the revised experiments ({NSplits} splits), which replace the earlier 5-split results.

---

## Critical issues

### 1. Synthetic data versus reported numerical results (reviewer §2)

> *"The authors must choose one of two scientifically defensible routes."*

**We chose Route B.** The public CSV could not be obtained in the computing environment used for this study, so the
paper is now explicitly a synthetic-data methodological study.

- The contradictory sentence ("All tables and figures are regenerated from the original data…") has been removed.
- The title, abstract (Methods and Conclusions), Introduction (paragraph 4), a new Section 3.2, the opening of
  Section 4, the Limitations, and the Data-availability statement all state that **every numerical result comes
  from the synthetic surrogate**. The paper no longer contains any wording implying that the numbers describe the
  public dataset.
- The figure captions and table captions state "synthetic surrogate" where relevant.

Route A (re-running on the public data) remains a single command in the released code. We present it as future
work rather than claiming it.

### 2. The CNN is not a convincing contribution (§3)

**Agreed; the paper has been reframed.** The CNN is now one of eight models in a reproducibility analysis. The paper
states that logistic regression reproduces the label without error on every split (accuracy {LRAcc} ± {LRAccStd}),
that the CNN ({CNNAcc} ± {CNNAccStd}) does not improve on it, and that a learned model offers no practical advantage
over direct scoring for this label (Abstract; Discussion, "Role of the CNN").

### 3. CNN parameter count (§4)

**We re-checked this, and the reported count of {CNNParams} is correct.** The difference comes from the pooling
output length. Keras `MaxPooling1D(2)` with the default `valid` padding uses floor rounding, so 29 → 14 positions,
not 15:

| Layer | Output | Parameters |
|---|---|---|
| Conv1D(32, k=3) | 29 × 32 | 3·1·32 + 32 = 128 |
| Conv1D(64, k=3) | 29 × 64 | 3·32·64 + 64 = 6,208 |
| MaxPool1D(2) | 14 × 64 | 0 |
| Conv1D(64, k=3) | 14 × 64 | 3·64·64 + 64 = 12,352 |
| Flatten | 896 | 0 |
| Dense(64) | 64 | 896·64 + 64 = 57,408 |
| Dense(1) | 1 | 65 |
| **Total** | | **76,161** |

As requested, the paper now includes this layer-by-layer table (new Table 3). It is generated directly from the
executable model (`model.summary()`), so the paper and the code cannot disagree. Section 3.5 notes the floor
rounding.

### 4. Specification of the synthetic generator (§5)

**Added in full** (new Section 3.2 and Table 1). The section specifies:
- the latent-status probability (0.69);
- per-toddler, per-item response probabilities (U(0.45, 0.85) if latent-positive, U(0.03, 0.30) otherwise);
- conditional independence of the items given those probabilities;
- the conditional probabilities of sex, jaundice and family history;
- the independent distributions of age, ethnicity and respondent;
- that the label is computed from the items with the published rule, not from the latent status;
- that the surrogate was generated once (seed 42) and shared by all splits.

We also state which properties the generator reproduces (the deterministic item–label rule, class balance) and
which it does not (real inter-item correlations and demographic associations). We note explicitly that the
demographic-feature results reflect our parameter choices.

### 5. Clinical framing and title (§7, §8)

- **Title changed.** It follows the reviewer's Option 1 and adds "synthetic-data study".
- **Terminology.** The paper now uses *threshold-defined screening label* for the target. "Diagnosis" is reserved
  for clinician-confirmed status, and the Introduction defines this distinction.
- The fact that the label is not a clinical diagnosis now appears in the Abstract (Background), Introduction
  (paragraph 3), Section 3.1, Results and Conclusion.

---

## Major issues

### 6. Statistical rigour (§6)

- **Splits increased from 5 to {NSplits}.** Table 2 now reports mean ± SD and the 2.5th–97.5th percentile range of
  accuracy across splits. We state that the splits overlap, so the SD describes split-to-split variability rather
  than a standard error. The number of unique test records covered ({NUniqueTest} of {NRecords}) is reported.
- **Confidence intervals.** The new Table 4 gives Wilson 95% CIs for sensitivity and specificity on the held-out E1
  split, with its size stated ({NTest} records: {NTestPos} positive, {NTestNeg} negative).
- **Paired comparisons.** The new Table 4 also gives exact McNemar tests of every model against logistic regression
  on the same test records.
- **DeLong's test** was considered but not used, because several AUCs equal 1.000 and the test is uninformative
  there. Section 3.6 explains this.

### 7. Independent clinical endpoint (§7)

Moved forward and emphasised (see item 5). The Discussion now recommends validation against ADOS-2 or
multidisciplinary consensus in a realistic-prevalence population.

### 8. Ablation interpretation (§10)

We added: "*Under the generator, demographic attributes depend on the latent status only weakly and on the label only
indirectly. This result therefore says nothing about the clinical value of these attributes.*" (Section 4.4.)

### 9. Permutation importance (§11)

- Permutations increased from 10 to {NPerm} per variable.
- Evaluation now covers {NImpSplits} splits instead of one, and error bars (SD) are shown.
- Logistic regression was added to the importance analysis.
- A new panel (Fig. 8b) shows the logistic-regression coefficients. The item coefficients lie between {ItemCoefMin}
  and {ItemCoefMax}, close to the scoring rule's equal weights.
- The caveat that the rankings are not a clinical ordering has been kept and made more prominent.

### 10. Clinical metrics (§12)

The new Table 4 reports sensitivity, specificity, PPV, NPV, LR+ and LR−. It also reports PPV re-weighted to a 3%
population prevalence, from both the point estimates and the lower confidence bounds. This produced a new finding:
even for the error-free logistic regression, the lower specificity bound ({LRSpecLo} on {NTestNeg} negatives)
implies a PPV as low as **{LRPPVpopLo}** at 3% prevalence. The prevalence figure now uses the most recent US
surveillance (1 in 31, 2022; Shaw et al., 2025). Calibration and subgroup analyses are listed as future work,
because they are not meaningful on this synthetic label.

### 11. Reproducibility details (§13, §17)

Section 3.5 and the new "Computational environment" paragraph now report:
- Software versions: Python {EnvPython}, TensorFlow {EnvTF}, scikit-learn {EnvSklearn}, NumPy {EnvNumpy},
  pandas {EnvPandas}.
- Hardware: CPU only ({EnvCPU}, {EnvNCPU} cores).
- Seeds for every experiment, and that TensorFlow's deterministic-operation mode is enabled.
- The initialisation scheme (Glorot-uniform kernels, zero biases).
- The validation split. It is now class-stratified (changed from Keras `validation_split`, which takes the last
  15% unstratified) and drawn per run with the run's seed.
- The number of CNN trainings: {NSplits} in E2, 3 × {NSplits} in E3 and {NInits} in E5.

The repository README separates the synthetic-data command from the public-data command. It also warns that the
paper's text describes the synthetic surrogate and would need rewriting after a public-data run. A single command
regenerates every table, figure and in-text number (`experiments.py` writes `generated/macros.tex`).

### 12. Figures (§14)

- **Fig. 6 (ROC):** a zoomed inset (FPR ≤ 0.25, TPR ≥ 0.75) was added.
- **Fig. 7 (learning curves):** now mean ± SD over {NInits} initialisations (new experiment E5). CNN accuracy across
  initialisations was {InitAccMean} ± {InitAccStd}.
- **Fig. 8 (importance):** now has error bars and a coefficient panel.

### 13. Wording (§15)

- "the less costly type in screening" was replaced with: *"False positives and false negatives both carry costs.
  Missed cases delay intervention, while false positives add family burden and demand on assessment services.
  Operating points should therefore be chosen for the intended setting."*
- "LR should be preferred" was replaced with a statement that LR "matches its performance with about 30 parameters
  instead of {CNNParams}, trains in milliseconds, and exposes coefficients that can be checked directly against the
  scoring rule".

---

## Minor issues

### 14. Reference verification (§16)

Every reference was checked against the publisher or an indexing database, and DOIs were added where they exist.
The check found three issues, which have been corrected:
1. **Hosseini et al. (2022), *Front. Comput. Neurosci.*, was retracted in May 2023.** It has been removed.
2. **Thabtah (2019) and Thabtah & Peebles (2020), *Health Informatics Journal*, are the subject of a 2021 expression
   of concern** about their peer review. We now disclose this in Related Work, Limitations and the Ethics statement,
   and cite the notice. We cite these articles only as the source of the dataset.
3. **The prevalence citation was updated** from Maenner et al. (2023; 1 in 36, 2020 data) to Shaw et al. (2025;
   1 in 31, 2022 data).

**Q-CHAT-10 cut-off.** Allison et al. (2012) report a cut-point of 3 (sensitivity 0.91, specificity 0.89, PPV 0.58 in
126 toddler cases and 754 controls), and the published instrument advises considering referral for scores *more
than 3*. The dataset's rule (score > 3) matches the instrument, and the paper now states both.

### 15. Ethics (§18)

The statement now explains that only synthetic data were used and names the public dataset's source. It also notes
that the descriptor articles do not describe the consent or ethics procedures of the original collection, and that
those articles are under an expression of concern.

### 16. Other additions

- A funding statement.
- A declaration of generative-AI assistance, as required by Elsevier. It states that Claude Code was used to draft
  the software, run the experiments and draft the text, and that the author takes full responsibility for the
  content.
