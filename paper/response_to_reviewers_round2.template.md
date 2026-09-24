# Response to the reviewer (round 2)

**Manuscript:** *Reproducibility and label leakage in machine learning on Q-CHAT-10 toddler screening data: a synthetic
benchmark of a 1D CNN and classical classifiers* (title revised as the reviewer suggested)

We thank the reviewer for a second careful review and for recognising the improvements. We have addressed all 14
priority items; each is answered below. All numbers are from the regenerated experiments.

---

## Must-fix items

### 1. Direct Q-CHAT-10 scoring as an explicit baseline (§7)

**Added.** The published rule applied with no learning, ŷ = 1[Σ A_j > 3], is now the first row of the main results
(Table 3), the screening-metrics table (Table 4), the confusion matrices (Fig. 5) and the ROC curves (Fig. 6). Its
ROC score is the item sum. All McNemar tests now compare each model with this rule rather than with logistic
regression. Results over {NSplits} splits: the rule scores accuracy {RuleAcc} and AUC {RuleAUC}, as expected.
Logistic regression matches it on every split ({LRAcc} ± {LRAccStd}; McNemar p = {LRMcNemarP} on E1). The CNN
reaches {CNNAcc} ± {CNNAccStd} (p = {CNNMcNemarP}). The Abstract, Discussion and Conclusion now state that neither
learned model improves on direct scoring.

### 2. "20 independent splits" (§9)

**Corrected.** The protocol now reads "the direct rule and all eight models on {NSplits} repeated stratified 80/20
holdout splits of the same benchmark (seeds 0–19). These are repeated random partitions, not independent datasets."
The word "independent" no longer describes the splits anywhere in the paper.

### 3. Does E5 include the seed-42 run? (§10)

**It does not, and the paper now says so explicitly.** E5 uses seeds {InitSeedMin}–{InitSeedMax}, which set both
the initialisation and the validation subset. The E1 run uses seed 42 and is not one of the ten. The seed-42 run
made {CNNEOneErrors} errors on the E1 test set, while the ten E5 runs scored {InitAccMean} ± {InitAccStd}. Both
facts are reported in Section 4.3, and the seeds are listed in the new Appendix A.

### 4. "A linear model represents this function exactly" (§5)

**Reworded** as the reviewer suggested: "On binary item inputs, a linear decision boundary can reproduce this
threshold classification exactly. This lets LR match the direct rule on every split…"

### 5. "All predictive information lies in the ten items" (§4)

**Narrowed** throughout:
- **Conclusion:** "For this label, the ten items contain essentially all the information needed to reproduce it.
  This says nothing about the value of other attributes for clinical endpoints."
- **Abstract:** "near chance for this label".
- **Introduction contributions:** "which inputs the models use to reproduce this label".
- **Section 4.4:** "for this threshold-defined label".

### 6. The 3% prevalence is illustrative (§8)

**Clarified, and a sensitivity analysis added.** Section 3.6 now says the prevalences are illustrative. It notes
that 3% is close to the prevalence of identified ASD among eight-year-olds, and that prevalence among toddlers
presenting for screening depends on the setting.

The new Table 5 reports PPV at 1%, 3%, 5% and 10% prevalence, from the E1 point estimates and from the lower Wilson
bounds. It also includes a reference row computed from the instrument's published accuracy (sensitivity 0.91,
specificity 0.89; Allison et al., 2012):

| Prevalence | Rule / LR, value at lower bounds | Published Q-CHAT-10 |
|---|---|---|
| 1% | {PPVRuleLoOne} | {PPVPubOne} |
| 3% | {LRPPVpopLo} | {PPVPubThree} |
| 10% | {PPVRuleLoTen} | {PPVPubTen} |

### 7. CNN feature-order sensitivity (§12)

**Added as experiment E6** (Section 4.4, Table 6). The CNN was retrained on the first {NOrderSplits} splits with
the columns in the original order (items first), reversed (items last), and in {NRandOrders} random permutations
with fixed seeds. Mean accuracy was {OrderOrigAcc} for the original order, {OrderRevAcc} reversed, and
{OrderRandAcc} ± {OrderRandAccStd} for the random orders (minimum {OrderRandAccMin}). The CNN's result therefore
does not depend on item adjacency. Section 3.5 now notes that the non-item variables have no natural order, and
the Discussion states that E6 gives no support to the idea that convolution over item neighbourhoods drives the
result.

### 8. Synthetic *benchmark*, not *surrogate* (§3, §20.8)

**Adopted throughout.** The paper now uses "synthetic benchmark constructed to reproduce the public dataset's schema,
class balance and deterministic label rule". Section 3.2 adds: "It is a benchmark built to reproduce these
structural properties, not a statistical surrogate of the public data's joint distribution." The Introduction
states: "It is not a validation study on the public dataset."

---

## Strongly recommended items

### 9. PPV sensitivity analysis
See item 6 (new Table 5).

### 10. Exact random seeds
The new **Appendix A, Table A.1** lists the seed for each component:
- benchmark generator and E1: 42;
- E2 and E3 splits: 0–19;
- E4 permutations: 0;
- E5: {InitSeedMin}–{InitSeedMax};
- E6 splits: 0–4, and column permutations: 100–102.

### 11. Machine-readable results
`paper/generated/results_summary.json` holds every result (E1 screening metrics, PPV by prevalence, E2 and E3
means and SDs, E4 importance and coefficients, E5, E6) along with the seeds and environment. Per-split CSVs are also
provided for E1–E6.

### 12. Environment file
`requirements-lock.txt` pins the exact versions used (Python {EnvPython}, TensorFlow {EnvTF}, scikit-learn
{EnvSklearn}, NumPy {EnvNumpy}, pandas {EnvPandas}, SciPy, Keras, Matplotlib). `requirements.txt` keeps the
minimum versions.

### 13. Exact reproduction command
Appendix A gives the three commands used to produce the paper. The Computational-environment paragraph names the
lock file and the results file.

### 14. Rule-based baseline in Table 3
Done; see item 1.

---

## Other comments

- **§13 (artificially easy task):** a new Discussion paragraph, "What this benchmark cannot establish", lists
  generalisation, robustness to distribution shift, discrimination of clinically diagnosed ASD, calibration and
  real-world screening utility.
- **§14 (provenance):** a new Related-work paragraph, "Scope of the evidence", separates four things: dataset
  provenance, the claims of the original articles (under an expression of concern), independent validation of the
  dataset (none known), and the present synthetic experiment.
- **§15 (ethics):** reworded as suggested: "Because the experiments used only author-generated synthetic data, no
  human-participant research was conducted in this study."
- **§16 (AI declaration):** added: "The released code and its outputs, not the AI-assisted workflow, are the
  authoritative record of the computational results. Every reported number can be regenerated independently with
  the command in Appendix A."
- **§18 (speculative phrasing):** replaced with "Evaluating CNNs would be more informative with richer
  longitudinal or multimodal inputs … and an independently established clinical endpoint."
- **§19 (title):** the suggested alternative title has been adopted.
