# Model Card: BBO Capstone Optimisation Approach

**Imperial College London / Emeritus PCMLAI Programme**
**Author:** Steven Suarez | **Last updated:** Round 10

---

## Overview

| Field | Detail |
|-------|--------|
| **Name** | Score-Guided Surrogate BBO Strategy |
| **Type** | Iterative black-box optimisation with neural network surrogate |
| **Version** | Round 10 (final programme version) |
| **Repository** | https://github.com/stevenasuarez/imperial-capstone-project |
| **Entry point** | `python src/main.py --round N` |

---

## Intended Use

**Suitable for:**
- Iterative optimisation of unknown scalar functions where only input-output pairs are observable
- Settings where the query budget is extremely limited (one query per function per round)
- Educational demonstration of surrogate-assisted BBO strategy evolution
- Benchmarking simple neural network surrogates against heuristic baselines

**Use cases to avoid:**
- Functions with known structure — gradient-based or analytical methods are more efficient
- High-frequency real-time optimisation — the strategy requires retraining per round
- Settings requiring uncertainty quantification — the current surrogate provides point estimates only, not confidence intervals
- Functions with discontinuities or multimodal surfaces — the gradient-based surrogate assumes local smoothness

---

## Strategy Details — Evolution Across Ten Rounds

**Round 1 — Heuristic initialisation:**
Queries placed by intuition across the input domain. No model. Purpose: establish a starting point before any feedback is available.

**Round 2 — Directional refinement:**
Manual coordinate adjustments based on spatial reasoning. Conservative nudges for low-dimensional functions, larger jumps for high-dimensional ones.

**Round 3 — SVM-guided search:**
Linear SVM trained on two labelled points per function. The weight vector (normal to the decision boundary) was used as the search direction. `step_scale=0.5`.

**Round 4 — Numpy NN surrogate with manual backpropagation:**
Three-layer fully-connected network (16→8→1) built from scratch in numpy. Synthetic labels (0.0, 0.5, 1.0). Manual chain rule backpropagation to compute input gradients. `step_scale=0.4`.

**Round 5 — PyTorch surrogate + autograd:**
Upgraded to PyTorch (32→16→8→1). Automatic differentiation replaces manual backpropagation. Adam optimiser. `step_scale=0.35`.

**Round 6 — Dimension-aware pooling (CNN-inspired):**
Activity mask computed from historical coordinate movement. High-activity dimensions amplified, low-activity dimensions dampened — analogous to CNN max-pooling over the time axis. Boundary correction introduced. `step_scale=0.3`.

**Round 7 — Hyperparameter grid search with LOO-CV:**
Grid search over learning rate [0.001, 0.005, 0.01, 0.05] and hidden size [16, 32]. Leave-one-out cross-validation selects best configuration per function. Step scale auto-selected from [0.2, 0.3, 0.4].

**Round 8 — Transformer attention-weighted gradient:**
Scaled dot-product attention `softmax(QK^T / sqrt(d))` applied across the round history. Current query is Q; previous rounds are K. Attention-weighted historical direction blended 40/60 with surrogate gradient. `step_scale=0.25`.

**Round 9 — Real score-guided strategy:**
Oracle scores received for Rounds 3–8. Synthetic labelling abandoned. Per-function strategy based on score trends: exploit (f5), continue (f2, f7, f8), reset (f1, f4), reverse (f3, f6).

**Round 10 — Real-score surrogate + interpretable decisions:**
Surrogate trained on real normalised oracle scores. Every decision documented with explicit reasoning printed at runtime. Per-function strategies differentiated by trend analysis across seven weeks of real data.

---

## Performance

**Score summary across functions (best observed score, week achieved):**

| Function | Best Score | Week | Current Trend | Status |
|----------|-----------|------|---------------|--------|
| f1 | ~2.68e-9 | W9 | ~0 everywhere | ❌ No viable region found |
| f2 | 0.7247 | W3 | Noisy, 0.57 at W9 | 🟡 Moderate |
| f3 | −0.0585 | W9 | Improving after reset | 🟡 Recovering |
| f4 | −3.986 | W9 | Improving after reset | 🟡 Recovering |
| f5 | 1623.03 | W5 | 1189 at W9, recovering | 🟢 High value |
| f6 | −0.5522 | W3 | Fluctuating negative | 🔴 Negative |
| f7 | 1.1880 | W4 | Stable 1.184 | 🟢 Stable positive |
| f8 | 8.0724 | W4 | Stable 8.059 | 🟢 Stable positive |

**Metric:** Oracle scalar output (maximisation objective). No normalisation applied to reported scores — raw oracle values used for comparability.

**Key finding:** f5 dominates all other functions by two orders of magnitude. The strategy correctly identified the f5 peak region at Week 5 but drifted away from it in Rounds 6–8 due to synthetic label bias, recovering in Rounds 9–10 after real scores became available.

---

## Assumptions and Limitations

**Core assumptions:**

1. **Monotonic improvement assumption (Rounds 3–8):** Synthetic labels assumed each round was strictly better than the previous. This was incorrect for most functions and introduced systematic directional errors into the surrogate for five rounds.

2. **Function stationarity:** Returning to previously high-scoring coordinates is assumed to reproduce similar scores. If the oracle has a stochastic component, this assumption may not hold.

3. **Local smoothness:** The neural network surrogate assumes the response surface is smooth enough for gradient-based navigation to be meaningful. Functions with sharp discontinuities or narrow peaks (likely f1) violate this assumption.

4. **Progressive improvement:** The score-guided strategy from Round 9 onward assumes that the best observed score is a reliable indicator of the best reachable score in the neighbourhood, which may miss better global regions.

**Key limitations:**

- **No uncertainty quantification:** The surrogate provides point estimates only. A Gaussian Process surrogate (e.g. GPyTorch) would provide posterior uncertainty, enabling principled exploration of high-uncertainty regions — the standard approach in production Bayesian optimisation.
- **Sparse data:** 10 points per function is insufficient to reliably characterise response surfaces, especially in 6D (f7) and 8D (f8).
- **f1 remains unsolved:** Near-zero scores across all ten rounds suggest the function peak is in a narrow region not yet sampled. The current gradient-based approach cannot detect it without first landing near it.
- **Boundary issues:** Several functions repeatedly hit [0,1] boundaries (f3, f5), distorting gradient estimates near the edges of the input domain.

---

## Ethical Considerations

**Transparency and reproducibility:**
The strategy is fully documented and reproducible. Every round's query coordinates are stored in `data_loader.py`, the complete code with inline reasoning is in `main.py`, and the runtime output explicitly prints the score trend, best historical week, and decision rationale for each function. Any researcher with access to the repository can reproduce all queries and understand exactly why each coordinate was chosen.

**Limitations of transparency:**
Transparency in documentation does not eliminate the underlying epistemic problem: for Rounds 3–8, the strategy was confident in directions that real scores later revealed to be wrong. Documenting a flawed decision process clearly is valuable for learning but does not make the decisions correct. This is an important lesson for real-world ML deployment: interpretability and transparency are necessary but not sufficient conditions for trustworthy systems.

**Broader applicability:**
The BBO capstone mirrors real-world scenarios — drug discovery, materials optimisation, engineering design — where each evaluation is expensive, feedback is delayed, and ground truth is unavailable during the search. The discipline of documenting assumptions, flagging when they were violated, and updating the strategy when evidence contradicts them is directly transferable to these high-stakes domains.
