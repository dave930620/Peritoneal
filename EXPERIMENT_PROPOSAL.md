# Experimental Proposal for IEEE J-BHI Submission
## Two-Stage PD Kt/V Prediction and Prescription Optimization

---

## 1. System Overview (for context)

Your pipeline has **two distinct models**, and each must be evaluated independently
**and** together:

| Model | Role | Architecture | Output |
|-------|------|--------------|--------|
| **F1 (SAINT)** | Adequacy predictor | Dual-embedding + Transformer encoder | PD Kt/V (continuous) |
| **F2 (Two-stage)** | Prescription optimizer | Stage A: CatBoost mimic → Stage B: MLP with F1 oracle | Rx variables (8 cont. + 1 cat.) |

The core novelty the paper must defend:
1. **F1**: Dual-path embedding (separate discrete vs. continuous projections) → single token → Transformer, applied to PD adequacy prediction.
2. **F2**: Differentiable surrogate optimization anchored by a CatBoost "doctor mimic" teacher, with PASS/FAIL-gated loss, clinical trust-region projection, and curriculum-scheduled constraint tightening.

---

## 2. Baselines

### 2.1 F1 Baselines — PD Kt/V Regression + Adequacy Classification

These are direct competitors to your SAINT model. All must be trained on the same
patient-wise 80/20 split, same features, same random seed.

| # | Model | Recency | Justification |
|---|-------|---------|---------------|
| B1 | **Ridge Regression** | Classical | Linear ceiling; shows whether the problem is even non-linear |
| B2 | **Random Forest** | Classical | Strong non-linear baseline, handles mixed types |
| B3 | **XGBoost** | 2016, still SOTA on tabular | State-of-the-art gradient boosting benchmark |
| B4 | **CatBoost** | 2018 | Best out-of-box handling of categorical features; directly comparable since you use it in F2 Stage A |
| B5 | **Vanilla MLP** | — | Ablation anchor: Transformer vs. plain feedforward |
| B6 | **TabNet** | Arik & Pfister, AAAI 2021 | Attention-based tabular, sequential feature selection |
| B7 | **FT-Transformer** | Gorishniy et al., NeurIPS 2021 | Most direct competitor — feature tokenizer + Transformer, similar to SAINT |
| B8 | **SAINT (original paper)** | Somepalli et al., 2021 | If you deviate from the original architecture (no full intersample attention), explicitly compare to the canonical version |

> **Why these?** J-BHI reviewers will look for at least one 2021+ deep-learning
> tabular baseline. FT-Transformer (B7) is the required "recent DL" reference.
> CatBoost (B4) is mandatory because you use it in F2 — reviewers will ask why not
> just use CatBoost for F1 too.

---

### 2.2 F2 Baselines — Prescription Optimization

F2 has no single "ground truth optimal" prescription, so baselines must be evaluated
by what F1 predicts the generated Rx will achieve, while also checking clinical
plausibility against the doctor's recorded prescription.

| # | Model | Description |
|---|-------|-------------|
| C1 | **Doctor's current prescription** | The actual recorded Rx — the floor and reference point |
| C2 | **Stage A only (CatBoost mimic)** | Output of Stage A without any MLP optimization; shows what pure imitation achieves |
| C3 | **Unconstrained MLP** | Stage B MLP trained without trust-region constraint (λ_constraint = 0, no projection); shows if the constraint hurts or helps |
| C4 | **Rule-based grid search** | For each FAIL patient: grid-search Rx space within STEP_MAP bounds, pick the Rx maximizing F1's predicted Kt/V; computationally expensive but interpretable upper bound |
| C5 | **Direct end-to-end MLP (no Stage A)** | MLP trained directly to maximize F1's output with no CatBoost teacher anchor; shows the value of the two-stage design |

> **Key question C2 answers:** Does the MLP optimization (Stage B) actually
> improve over just mimicking the doctor? If C1 (doctor) ≈ C2 (mimic) < your model,
> you have a complete story.

---

## 3. Metrics

### 3.1 F1 Metrics

Report **all** of the following. J-BHI expects both regression and clinical metrics.

**Regression metrics** (primary — all test set):
| Metric | Why |
|--------|-----|
| MAE (mean absolute error) | Clinical interpretability: "off by X Kt/V on average" |
| RMSE | Penalizes large errors; standard regression metric |
| R² | Proportion of variance explained |
| Pearson r | Linear correlation |

**Threshold-based classification metrics** (at Kt/V = 1.7):

Each regression model's continuous output is thresholded at 1.7 to produce
binary "adequate / inadequate" predictions, then:

| Metric | Why |
|--------|-----|
| AUROC | Discrimination ability; threshold-agnostic |
| AUPRC | Important when PASS/FAIL may be imbalanced |
| Accuracy | Simple clinical communication |
| Sensitivity (Recall) | Catching inadequate patients — clinically critical |
| Specificity | Avoiding false alarms |
| F1-score (binary) | Balances precision and recall |

> **Reporting format**: One table with all 10 metrics × all 8 baselines + your model.
> Bold the best per column. If your model wins 7/10, that is a strong result.

---

### 3.2 F2 Metrics

Two categories: **optimization effectiveness** and **clinical safety**.

**Optimization effectiveness** (F1 model evaluates the generated Rx):
| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| Mean ΔKt/V (FAIL group) | `mean(F1(model_Rx) - F1(doctor_Rx))` for patients where doctor Kt/V < 1.7 | How much the model improves the worst-off patients |
| P_pass (FAIL group) | `P(F1(model_Rx) ≥ 1.7 | doctor_Rx < 1.7)` | Proportion of previously failing patients pushed above threshold |
| Mean ΔKt/V (PASS group) | `mean(F1(model_Rx) - F1(doctor_Rx))` for patients where doctor Kt/V ≥ 1.7 | Confirms no regression in patients who were already adequate |
| P_fail (PASS group) | `P(F1(model_Rx) < 1.7 | doctor_Rx ≥ 1.7)` | Safety: does the model accidentally break passing patients? |

**Clinical plausibility / safety** (how close to doctor's Rx):
| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| Pearson r (per continuous Rx) | Between model Rx and doctor Rx for each of 8 cont. variables | How faithful is the optimizer to clinical practice |
| Categorical accuracy | `mean(model CAT_RX == doctor CAT_RX)` | Agreement on PD system type |
| Mean absolute Rx deviation | `mean(|model_Rx - doctor_Rx|)` per variable | Practical magnitude of suggested change |

> **Reporting format**: Table split into FAIL group / PASS group rows, with each
> baseline (C1–C5) in columns.

---

## 4. Ablation Studies

Ablations justify **every design choice** — this is exactly what the professor
warned about ("clearly justify every methodological choice").

### 4.1 F1 Ablations — Justify the SAINT Architecture

Run each ablation on the same split/seed as your full model.
Report all 10 metrics from §3.1.

| ID | What is removed / changed | What it tests |
|----|---------------------------|---------------|
| **A-F1-1** | Replace dual-path embedding with single `nn.Linear(input_size, hidden_size)` | Is the discrete/continuous split beneficial? |
| **A-F1-2** | Remove `nn.LayerNorm` from embedding blocks | Is normalization in the embedding necessary? |
| **A-F1-3** | Replace `nn.GELU` with `nn.ReLU` in embedding + output head | Activation function choice |
| **A-F1-4** | Replace Transformer encoder with a 3-layer MLP (same hidden size) | Is self-attention the key contributor or just depth? |
| **A-F1-5** | num_layers = 2 (vs. 6) | Depth sensitivity |
| **A-F1-6** | num_layers = 4 (vs. 6) | Depth sensitivity |
| **A-F1-7** | num_heads = 1 (vs. 8) | Multi-head attention necessity |
| **A-F1-8** | num_heads = 4 (vs. 8) | Multi-head attention sensitivity |
| **A-F1-9** | Remove `ReduceLROnPlateau` scheduler (constant lr) | LR scheduling contribution |
| **A-F1-10** | Remove AdamW weight decay (set to 0) | Regularization contribution |

> **Minimum ablations for J-BHI**: A-F1-1 (dual embedding), A-F1-4 (Transformer
> vs. MLP), A-F1-5/6 (depth), A-F1-7/8 (heads). These directly defend novelty.

---

### 4.2 F2 Ablations — Justify Every Loss Term and Design Choice

Report all F2 metrics from §3.2 for each ablation variant.

| ID | What is removed / changed | What it tests |
|----|---------------------------|---------------|
| **A-F2-1** | Remove Stage A entirely; train MLP from scratch with no teacher anchor (λ_prox_cont = λ_prox_cat = 0) | Does the two-stage design (CatBoost teacher) matter? |
| **A-F2-2** | Remove trust-region constraint (λ_constraint = 0, no projection in `infer_prescriptions`) | Does the hinge penalty prevent out-of-distribution Rx? |
| **A-F2-3** | Remove threshold push loss (λ_thr_fail = 0) | Does explicitly pushing below-threshold cases matter? |
| **A-F2-4** | Remove PASS/FAIL gating — use uniform weights (w_effect_pass = w_effect_fail, w_thr_pass = w_thr_fail) | Is the asymmetric treatment of FAIL patients necessary? |
| **A-F2-5** | Remove curriculum scheduling — set lc = 1.0 throughout training | Does constraint curriculum matter? |
| **A-F2-6** | Remove proximal loss to teacher (λ_prox_cont = λ_prox_cat = 0, but keep Stage A for evaluation reference) | Is the proximal anchor term specifically needed? |
| **A-F2-7** | Replace MLP (F2RxHead) with linear head (remove backbone, direct linear mapping) | Is non-linearity in the optimizer necessary? |
| **A-F2-8** | Expand trust region for PASS cases to match FAIL (EPS_CONT_Z_PASS = EPS_CONT_Z_FAIL = 3.0) | Is the asymmetric trust region important? |

> **Minimum ablations for J-BHI**: A-F2-1 (two-stage design), A-F2-2 (trust
> region), A-F2-3 (threshold loss), A-F2-4 (PASS/FAIL gating). These are the
> four novel design choices that reviewers will challenge.

---

## 5. Comparison Strategy: Per-Model vs. End-to-End

You asked whether to compare end-to-end, F1 only, or F2 only.
**The answer is all three, in separate tables.**

```
Paper structure:

Table 1 — F1 Prediction  (F1 model vs. all F1 baselines B1–B8)
           Metric: MAE, RMSE, R², Pearson r, AUROC, AUPRC, Acc, Sens, Spec, F1

Table 2 — F2 Optimization  (your full F2 vs. baselines C1–C5)
           Metric: ΔKt/V, P_pass, P_fail, Pearson r per Rx var, cat. accuracy

Table 3 — F1 Ablations  (A-F1-1 through A-F1-10 vs. full SAINT)
Table 4 — F2 Ablations  (A-F2-1 through A-F2-8 vs. full F2)

Figure  — Interpretability (see Section 6)
```

**There is no single "end-to-end baseline"** because the tasks are fundamentally
different (F1 = supervised regression, F2 = constrained optimization). What you
do instead is note that **F2 depends on F1** and that the full system's clinical
benefit is captured by Table 2's P_pass / P_fail metrics.

---

## 6. Interpretability Analysis (Required for J-BHI)

The professor specifically called this out. Include the following:

### 6.1 F1 Interpretability
1. **Permutation feature importance** — already computed (`feature_importance.csv`).
   Report the top-15 features in a horizontal bar chart.
2. **SHAP values** — run `shap.Explainer` on your SAINT model (or approximate with
   kernel SHAP). Show:
   - Global SHAP summary plot (beeswarm)
   - Local SHAP waterfall for 2–3 representative patients (one FAIL, one PASS, one borderline)
3. **Attention weight visualization** — for the Transformer encoder, extract the
   attention weights from the last layer for a sample of patients and plot as a heatmap
   over feature indices. This shows *which features the model attends to*.

### 6.2 F2 Interpretability
1. **Prescription change analysis** — for the FAIL group, show a table of mean
   suggested Rx change vs. doctor's Rx per variable (e.g., "model suggests +0.4 bags/day
   on average for FAIL patients"). This is the most clinically meaningful result.
2. **CatBoost Stage A feature importance** — use `catboost_model.get_feature_importance()`
   for each Rx variable. Show which patient features drive each prescription.
3. **Case study** — pick 2–3 real patients, show doctor Rx → model Rx → predicted
   Kt/V improvement. This qualitative result is often the most memorable in clinical ML papers.

---

## 7. Statistical Significance

J-BHI reviewers will ask for this.

- For F1: Report **mean ± std** across **5-fold cross-validation** (patient-wise GroupKFold),
  not just a single 80/20 split. Run all baselines with the same folds.
- For F2: Report **bootstrap confidence intervals** (95%) on P_pass and ΔKt/V by
  resampling the test set 1000 times.
- Use **paired t-test** or **Wilcoxon signed-rank test** to compare your model vs.
  the best baseline per metric in Table 1.

---

## 8. Novelty Argumentation (for the paper's Introduction / Methodology)

The reviewer concern is that "combining off-the-shelf components" lacks novelty.
Here is how to frame each design choice as a novel contribution:

| Design Choice | "Off-the-shelf" framing (bad) | Novel framing (what to write) |
|---------------|-------------------------------|-------------------------------|
| SAINT for PD | "We use SAINT" | "We adapt SAINT's dual-path embedding to explicitly model the semantic distinction between binary clinical indicators (diagnosis flags) and continuous physiological measurements in PD records, which is architecturally justified by the heterogeneous feature space of dialysis data" |
| CatBoost + MLP two-stage | "We chain two models" | "We introduce a teacher-student architecture where a CatBoost model provides a safety anchor (doctor's distribution), enabling the MLP to explore prescription improvements without leaving clinically plausible Rx space" |
| Trust region | "We clip outputs" | "We formulate the trust region as a differentiable hinge penalty during training and a hard projection at inference, with asymmetric radii calibrated separately for adequate and inadequate patients" |
| PASS/FAIL gating | "We weight losses differently" | "We propose a clinical-state-conditioned loss weighting scheme that concentrates optimization pressure on inadequate patients (PD Kt/V < 1.7) while preserving the prescriptions of adequately-treated patients, directly operationalizing the clinical objective" |
| Curriculum scheduling | "We change lambda over epochs" | "We apply curriculum learning to constraint enforcement, tightening the trust region early in training to prevent mode collapse and relaxing it mid-training to encourage exploration near the adequacy boundary" |

---

## 9. Implementation Checklist

- [ ] Implement baselines B1–B8 using scikit-learn / PyTorch Tabular / tab-transformer
- [ ] Run 5-fold patient-wise GroupKFold for F1 (report mean ± std)
- [ ] Run all F2 baselines C1–C5 on the same patient-wise test split
- [ ] Run all ablations A-F1-1 through A-F1-10
- [ ] Run all ablations A-F2-1 through A-F2-8
- [ ] Add SHAP analysis for F1 (pip install shap)
- [ ] Add CatBoost feature importance for each Rx variable
- [ ] Write 2–3 patient case studies for qualitative analysis
- [ ] Compute bootstrap CIs on F2 test metrics
- [ ] Compute Wilcoxon test for F1 vs. best baseline
