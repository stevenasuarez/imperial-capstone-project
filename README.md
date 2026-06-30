Black-Box Optimisation Capstone

Imperial College London · Professional Certificate in Machine Learning & AI


For Everyone: What This Project Does

This project tackles a puzzle: how do you find the best settings for a system when you can't see inside it? Imagine trying to tune a recipe where you can only taste the result once a week, with no access to the ingredient list. Over thirteen weeks, I submitted one set of inputs per function to a scoring system and used the results to make smarter guesses each week. My strategy evolved from simple intuition to machine learning models guided by real feedback. The clearest win was Function 5, which reached a score of 1,668 — more than three times its starting value — by identifying which two of its four inputs matter most and pushing them toward their optimal values.


Documentation

DocumentDescriptionDATASHEET.mdDataset documentation — composition, collection process, gaps and terms of useMODEL_CARD.mdModel card — strategy overview, performance summary, assumptions and limitationsnotebooks/bbo_analysis.ipynbFull analysis notebook with visualisations and results


Section 1: Project Overview

This project tackles a Black-Box Optimisation (BBO) challenge — the task of finding the best inputs to a set of unknown functions without ever seeing their equations. The only feedback available is the output value returned by an oracle after each query is submitted.

The project spans eight unknown functions of varying dimensionality (2D through 8D). Over thirteen rounds, each query is one carefully chosen point in the input space. The challenge is to find high-performing regions using as few evaluations as possible.

Why does this matter in real-world ML?
BBO is everywhere: hyperparameter tuning for neural networks, drug dosage optimisation in clinical trials, A/B test design, and materials science experimentation all share the same structure — an expensive, opaque objective function that you can only probe one point at a time. Learning to search efficiently under these constraints is a core applied ML skill.

Career relevance:
As a QA Automation Engineer at a software enterprise, I work daily with systems whose behaviour is not always fully predictable — tests pass or fail based on complex interactions that are not always transparent. The BBO framework maps directly onto that reality: you probe a system, observe an outcome, and decide where to look next. The skills developed here — designing efficient experiments, interpreting signals with limited data, and iterating systematically — translate directly into smarter test strategy design and, longer term, into ML-assisted QA pipelines.


Section 2: Inputs and Outputs

Inputs — query format:
Each query is a vector of coordinates, one per dimension, where every value is a float in the range [0.0, 1.0] specified to six decimal places:

x1-x2-x3-...-xn

Example for a 3D function:

0.150000-0.220000-0.400000

Function dimensionalities:

FunctionDimensionsBest ScoreBest Weekf12~3×10⁻¹⁴W9f220.7247W3f33−0.047W10f44−3.986W9f541668.88W10f65−0.518W10f761.188W4f888.072W4

Outputs:
After each submission round, the oracle returns a single scalar performance value per function. Larger values are better — every function is a maximisation task. The internal structure is never revealed.


Section 3: Challenge Objectives

The goal is to maximise the output of each unknown function, subject to:


One query per function per round — no simultaneous evaluations
Inputs must lie in [0, 1] for every dimension
No access to gradients — derivative-free optimisation only
No knowledge of function structure — could be non-convex, discontinuous, or high-dimensional



Section 4: Technical Approach

Round 1 — Broad initialisation:
Spatial intuition, broad coverage. No model.

Round 2 — Directional refinement:
Manual coordinate adjustments. Conservative nudges for low-dimensional functions, larger jumps for high-dimensional ones.

Round 3 — SVM-guided search:
Linear SVM trained on two labelled points per function. Weight vector used as search direction. step_scale=0.5.

Round 4 — Numpy NN surrogate (manual backprop):
Three-layer network (16→8→1) built from scratch. Synthetic labels (0.0, 0.5, 1.0). Manual chain-rule backpropagation. step_scale=0.4.

Round 5 — PyTorch surrogate + autograd:
Upgraded to PyTorch (32→16→8→1). Automatic differentiation replaces manual backprop. Adam optimiser. step_scale=0.35.

Round 6 — Dimension-aware pooling (CNN-inspired):
Activity mask from historical coordinate movement amplifies active dimensions, dampens inactive ones. Boundary correction added. step_scale=0.3.

Round 7 — Hyperparameter grid search:
Grid search over learning rate [0.001, 0.005, 0.01, 0.05] and hidden size [16, 32]. Leave-one-out cross-validation selects best config per function.

Round 8 — Transformer attention-weighted gradient:
Scaled dot-product attention softmax(QK^T / sqrt(d)) over query history. Gradient blended 60/40 with attention-weighted historical direction. step_scale=0.25.

Round 9 — Real score-guided strategy (major upgrade):
Oracle scores received for Weeks 3–8. Synthetic labelling abandoned. Per-function strategy based on score trends:

FunctionScore trendStrategyf1~0 all roundsReset to centref2Best at W3Return toward best pointf3Consistently negativeReset toward centref4Getting worseHard reset to centref5Peaked W5, declinedReturn toward W5 coordinatesf6Consistently negativeReset toward centref7Stable positiveSmall continuation stepf8Stable positiveSmall continuation step

Round 10 — Real-score surrogate + interpretable decisions:
Surrogate trained on real normalised oracle scores. Every decision printed with explicit reasoning at runtime.

Round 11 — Clustering-based strategy:
Score-weighted centroid and high-score cluster centroid (top-3 points) computed per function. Queries stepped toward the most promising cluster.

Round 12 — PCA-guided search:
Principal components computed from scored query history. Score-aligned PC (highest correlation with oracle output) used as search direction. PC1 variance explained reported per function.

Round 13 — Final round: pure exploitation:
No surrogate, no gradient. Each function submits its single best historical point directly — the greedy argmax policy from Q-learning applied to the full score history.

Key insight from the full project:
The synthetic progressive labelling assumption (each round improves on the last) was incorrect for most functions. f5's best score occurred at Week 5 (1623) and the search drifted away for three rounds before real scores enabled recovery. Real feedback always supersedes model assumptions.


Usage

Requirements

bashpip install -r requirements.txt

Running a specific round

bashcd src
python main.py --round 3    # SVM-guided
python main.py --round 9    # Real score-guided
python main.py --round 13   # Final round
python main.py              # All rounds in sequence

Results saved automatically to results/queries_roundN.txt.

Project structure

Capstone/
├── src/
│   ├── main.py          # All round strategies (2-13)
│   └── data_loader.py   # All query coordinates rounds 1-13
├── results/
│   └── queries_round*.txt
├── notebooks/
│   └── bbo_analysis.ipynb
├── DATASHEET.md
├── MODEL_CARD.md
├── requirements.txt
└── README.md

Developed as part of the Imperial College London / Emeritus PCMLAI programme.