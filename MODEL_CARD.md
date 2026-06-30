Model Card: BBO Capstone Optimisation Approach

Imperial College London / Emeritus PCMLAI Programme
Author: Steven Suarez | Last updated: Round 13 (final)


Overview

FieldDetailNameScore-Guided Surrogate BBO StrategyTypeIterative black-box optimisation with neural network surrogateVersionRound 13 — FinalRepositoryhttps://github.com/stevenasuarez/imperial-capstone-projectEntry pointpython src/main.py --round N


Intended Use

Suitable for:


Iterative optimisation of unknown scalar functions where only input-output pairs are observable
Settings where the query budget is extremely limited (one evaluation per function per round)
Educational demonstration of surrogate-assisted BBO strategy evolution
Benchmarking simple neural network surrogates against heuristic baselines


Use cases to avoid:


Functions with known structure — gradient-based or analytical methods are more efficient
High-frequency real-time optimisation — strategy requires retraining per round
Settings requiring uncertainty quantification — current surrogate provides point estimates only
Functions with sharp discontinuities or multimodal surfaces at fine scale



Strategy Details — All Thirteen Rounds

RoundMethodKey concept1Heuristic initialisationBroad spatial coverage, no model2Directional refinementManual coordinate adjustment3Linear SVMDecision boundary weight vector as search direction4Numpy NN (manual backprop)Explicit chain-rule gradient computation5PyTorch + autogradAutomatic differentiation, Adam optimiser6Dimension-aware poolingCNN-inspired activity mask on gradient7Hyperparameter grid searchLOO cross-validation, tunable lr and architecture8Transformer attentionScaled dot-product attention over query history9Real score-guidedOracle scores replace synthetic labels entirely10Interpretable surrogateReal normalised scores, per-function reasoning printed11ClusteringScore-weighted centroid and high-score cluster centroid12PCA-guidedScore-aligned principal component as search direction13Pure exploitationGreedy argmax — best historical point per function


Performance

Score summary — best observed across all rounds:

FunctionBest ScoreWeekFinal Statusf1~2.68×10⁻⁹W9❌ Unsolved — near-zero everywheref20.7247W3🟡 Moderate — best early, recovered latef3−0.0472W10🟡 Negative but improvingf4−3.986W9🟡 Negative, improved significantly after resetf51668.88W10🟢 Dominant function, 3× initial valuef6−0.518W10🟡 Negative but best-ever after recoveryf71.1880W4🟢 Stable positive, near-optimalf88.0724W4🟢 Stable positive, near-optimal

Key finding:
f5 dominates all other functions by two orders of magnitude. The strategy correctly identified the f5 peak region at Week 5 but drifted away from it in Rounds 6–8 due to synthetic label bias, recovering to a new all-time best (1668) in Round 10 after real scores became available.

f7 and f8 converged to stable local optima by Round 4 and remained there throughout — PCA confirmed >88% of variance explained by a single tight cluster, indicating these functions were effectively solved early.

f1 remained near zero across all thirteen rounds. The function likely has an extremely narrow peak in a region never sampled. Gradient-based surrogates cannot locate such peaks without first landing near them.


Assumptions and Limitations

Core assumptions:


Monotonic improvement (Rounds 3–8): Synthetic labels assumed each round was strictly better than the previous. This was incorrect for most functions and introduced systematic directional errors for five rounds — most visibly for f5, which lost approximately 1,100 score units before correction.
Function stationarity: Returning to previously high-scoring coordinates is assumed to reproduce similar scores. If the oracle has any stochastic component this may not hold.
Local smoothness: The neural network surrogate assumes the response surface is smooth enough for gradient navigation. Functions with sharp discontinuities (likely f1) violate this.
Best observed = best reachable: The score-guided strategy assumes the historical best point is a reliable target. Better global regions may exist but were never sampled.


Key limitations:


No uncertainty quantification: The surrogate provides point estimates only. A Gaussian Process (GPyTorch) would provide posterior uncertainty enabling principled exploration of high-uncertainty regions.
Sparse data: 13 points per function is insufficient to reliably characterise response surfaces, especially in 6D (f7) and 8D (f8).
f1 unsolved: Near-zero scores across all thirteen rounds. Current gradient-based approach cannot locate a narrow peak without first landing near it.
Missing early scores: Rounds 1, 2 and 11 scores not available, limiting the completeness of the training history.



Ethical Considerations

Transparency and reproducibility:
The strategy is fully documented and reproducible. Every round's coordinates are stored in data_loader.py, the complete code with inline reasoning is in main.py, and the runtime output explicitly prints score trend, best historical week, and decision rationale for each function. Any researcher with repository access can reproduce all queries and understand exactly why each coordinate was chosen.

Limitations of transparency:
Documenting a flawed decision process clearly is valuable for learning but does not make the decisions correct. For Rounds 3–8, the strategy was confident in directions that real scores later revealed to be wrong. Interpretability and transparency are necessary but not sufficient conditions for trustworthy ML systems.

Broader applicability:
The BBO capstone mirrors real-world scenarios — drug discovery, materials optimisation, engineering design — where each evaluation is expensive, feedback is delayed, and ground truth is unavailable during the search. The discipline of documenting assumptions, flagging when they were violated, and updating the strategy when evidence contradicts them is directly transferable to these high-stakes domains.

The most important lesson:
Real feedback always supersedes model assumptions. A confident model trained on wrong labels is worse than no model at all. Invest in signal quality before model sophistication.