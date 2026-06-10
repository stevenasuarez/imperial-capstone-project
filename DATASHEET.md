# Datasheet: BBO Capstone Project Dataset

**Imperial College London / Emeritus PCMLAI Programme**
**Author:** Steven Suarez | **Last updated:** Round 10

---

## Motivation

**Why was this dataset created?**
This dataset was created as part of the Black-Box Optimisation (BBO) capstone project for the Imperial College London / Emeritus Professional Certificate in Machine Learning and AI. It documents the complete query history submitted to an oracle system across ten rounds of iterative optimisation, alongside the oracle's scalar output values for each query.

**What task does it support?**
The dataset supports the task of surrogate-assisted black-box optimisation: learning to maximise eight unknown functions by iteratively querying an oracle with candidate input vectors and using the returned scores to inform subsequent queries. It also serves as a teaching dataset for demonstrating the evolution of modelling strategies — from heuristic search to SVM-guided search, neural network surrogates, and score-driven decision-making.

**Who created it and for what purpose?**
Created by the project author as part of assessed coursework. The dataset is intended to document the full search trajectory, enable reproducibility of the strategy, and support reflection on decision quality across rounds.

---

## Composition

**What does the dataset contain?**
The dataset contains query vectors and oracle output scores for eight unknown functions across ten submission rounds (Rounds 1–10, corresponding to Weeks 1–10 of the programme).

**Format:**
Each record consists of:
- A function identifier (f1–f8)
- A query vector of floats in [0.0, 1.0] with six decimal places
- A scalar oracle output value (available for Weeks 3–10; Weeks 1–2 scores were not retained)

**Dimensionality per function:**

| Function | Input dimensions | Score range (observed) |
|----------|-----------------|----------------------|
| f1 | 2 | ~0 (near-zero everywhere) |
| f2 | 2 | 0.43 – 0.72 |
| f3 | 3 | −0.18 – −0.06 |
| f4 | 4 | −6.49 – −3.99 |
| f5 | 4 | 445 – 1623 |
| f6 | 5 | −0.76 – −0.55 |
| f7 | 6 | 1.184 – 1.188 |
| f8 | 8 | 8.059 – 8.072 |

**Size:** 80 query records (8 functions × 10 rounds), with oracle scores available for 64 records (Rounds 3–10).

**Gaps:**
- Oracle scores for Rounds 1 and 2 were not retained from the portal confirmation emails. These rounds used heuristic queries without model guidance and their absence does not materially affect the surrogate training, but limits the completeness of the historical record.
- f1 scores are functionally zero across all rounds (range: 8.76e-28 to 2.68e-9), suggesting the function has a near-zero response almost everywhere in the unit cube. The true peak region, if one exists, has not been found.

**Are there sensitive attributes?**
No. The dataset contains only numerical input-output pairs with no personally identifiable information.

---

## Collection Process

**How were queries generated?**
Queries were generated programmatically using a sequence of increasingly sophisticated strategies:

- **Rounds 1–2:** Manual heuristic initialisation — broad spatial coverage of the input domain
- **Round 3:** Linear SVM-guided search — decision boundary weight vector used as improvement direction
- **Round 4:** Numpy neural network surrogate with manual backpropagation
- **Round 5:** PyTorch surrogate with automatic differentiation (autograd)
- **Round 6:** Dimension-aware pooling — CNN-inspired activity mask on gradient
- **Round 7:** Hyperparameter grid search with leave-one-out cross-validation
- **Round 8:** Transformer-style scaled dot-product attention over query history
- **Rounds 9–10:** Real score-guided strategy using actual oracle outputs as surrogate labels

**What strategy was used?**
The strategy evolved from pure exploration (early rounds) toward exploitation of identified promising regions (later rounds), guided by progressive model upgrades. From Round 9 onward, per-function strategies were differentiated based on observed score trends: exploit, continue, recover, or reset.

**Over what time frame?**
One query per function per week over ten weeks (approximately ten weeks of the PCMLAI programme, Modules 12–21).

**Who submitted the queries?**
The project author, via the Imperial College / Emeritus BBO capstone portal.

---

## Preprocessing and Uses

**Have any transformations been applied?**

- Oracle scores were normalised to [0, 1] for surrogate training from Round 10 onward using min-max scaling across the observed score range per function.
- Input coordinates are already constrained to [0, 1] by the oracle's input specification — no additional input normalisation was applied.
- Synthetic progressive labels (0.0, 0.33, 0.67, 1.0, etc.) were used for surrogate training in Rounds 4–8 before real scores became available. These labels were an assumption, not ground truth, and introduced systematic bias for functions where the true score trajectory did not match the assumed progression.
- Boundary corrections were applied in Rounds 6–10: coordinates at or near 0.0 or 1.0 were pulled to 0.05/0.95 before gradient steps to avoid degenerate boundary behaviour.

**Intended uses:**
- Reproducibility and audit of the BBO search strategy
- Training and evaluating surrogate models for black-box optimisation
- Educational illustration of iterative query strategy design under uncertainty
- Benchmarking alternative BBO strategies against the recorded query trajectory

**Inappropriate uses:**
- Drawing conclusions about the true structure of the oracle functions — the dataset is too sparse (10 points per function) to reliably characterise the response surface
- Training production ML models — the dataset is too small and domain-specific
- Generalising findings to other BBO problems without accounting for the specific function characteristics observed here

---

## Distribution and Maintenance

**Where is the dataset available?**
The complete dataset is stored in the project's public GitHub repository:
[https://github.com/stevenasuarez/imperial-capstone-project](https://github.com/stevenasuarez/imperial-capstone-project)

Query records are stored in:
- `src/data_loader.py` — all round coordinates as Python dictionaries
- `results/queries_roundN.txt` — portal-format strings for each round

**Terms of use:**
This dataset was created for educational purposes as part of the Imperial College London / Emeritus PCMLAI programme. It may be freely used for academic and educational purposes with attribution. The oracle function outputs are the intellectual property of Imperial College London / Emeritus and are reproduced here solely for the purpose of documenting the project author's own submissions.

**Who maintains it?**
Steven Suarez. The dataset will be updated with each new round submission throughout the programme.
