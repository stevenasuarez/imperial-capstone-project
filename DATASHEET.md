Datasheet: BBO Capstone Project Dataset
Imperial College London / Emeritus PCMLAI Programme Author: Steven Suarez | Last updated: Round 13 (final)

Motivation
Why was this dataset created? This dataset was created as part of the Black-Box Optimisation (BBO) capstone project for the Imperial College London / Emeritus Professional Certificate in Machine Learning and AI. It documents the complete query history submitted to an oracle system across thirteen rounds of iterative optimisation, alongside the oracle's scalar output values for each query.

What task does it support? The dataset supports surrogate-assisted black-box optimisation: learning to maximise eight unknown functions by iteratively querying an oracle with candidate input vectors and using returned scores to inform subsequent queries. It also serves as a teaching dataset for demonstrating the evolution of modelling strategies — from heuristic search through SVM, neural network surrogates, clustering, PCA, and final pure exploitation.

Who created it and for what purpose? Created by Steven Suarez as part of assessed coursework. Intended to document the full search trajectory, enable reproducibility, and support reflection on decision quality across rounds.

Composition
What does the dataset contain? Query vectors and oracle output scores for eight unknown functions across thirteen submission rounds.

Format: Each record consists of:

Function identifier (f1–f8)
Query vector of floats in [0.0, 1.0] with six decimal places
Scalar oracle output value (available for Weeks 3–13)
Dimensionality and observed score ranges:

Function	Dimensions	Score range (observed)	Best score	Best week
f1	2	~0 everywhere	2.68×10⁻⁹	W9
f2	2	0.43 – 0.72	0.7247	W3
f3	3	−0.18 – −0.047	−0.047	W10
f4	4	−6.49 – −3.99	−3.986	W9
f5	4	445 – 1668	1668.88	W10
f6	5	−0.76 – −0.518	−0.518	W10
f7	6	1.182 – 1.188	1.1880	W4
f8	8	8.059 – 8.072	8.0724	W4
Size: 104 query records (8 functions × 13 rounds), with oracle scores available for 88 records (Rounds 3–13).

Gaps:

Oracle scores for Rounds 1 and 2 were not retained. These rounds used heuristic queries without model guidance.
Oracle scores for Round 11 were not retained (email lost). Values were interpolated between W10 and W12 for surrogate training purposes.
f1 scores are functionally zero across all rounds, suggesting an extremely narrow peak region not sampled by any query.
Sensitive attributes: None. Dataset contains only numerical input-output pairs.

Collection Process
How were queries generated?

Round	Strategy
1–2	Manual heuristic initialisation
3	Linear SVM — decision boundary weight vector
4	Numpy neural network — manual backpropagation
5	PyTorch surrogate — automatic differentiation
6	Dimension-aware pooling (CNN-inspired activity mask)
7	Hyperparameter grid search with LOO cross-validation
8	Transformer-style scaled dot-product attention
9–10	Real score-guided strategy (oracle scores as labels)
11	Score-weighted clustering centroid
12	PCA-aligned principal direction search
13	Pure exploitation — best historical point per function
Time frame: One query per function per week over thirteen weeks (Modules 12–24 of the PCMLAI programme).

Who submitted queries? Steven Suarez, via the Imperial College / Emeritus BBO capstone portal.

Preprocessing and Uses
Transformations applied:

Oracle scores normalised to [0, 1] using min-max scaling for surrogate training from Round 10 onward
Synthetic progressive labels (0.0 → 1.0) used for Rounds 4–8 before real scores were available — these introduced systematic bias and were replaced as soon as real scores arrived
Boundary corrections applied in Rounds 6–13: coordinates at or near 0.0/1.0 pulled to 0.05/0.95 before gradient steps
Round 11 scores interpolated between W10 and W12 due to missing email
Intended uses:

Reproducibility and audit of the BBO search strategy
Training and evaluating surrogate models for black-box optimisation
Educational illustration of iterative query strategy design under uncertainty
Benchmarking alternative BBO strategies against the recorded trajectory
Inappropriate uses:

Drawing conclusions about the true structure of the oracle functions — 13 points per function is insufficient to characterise the response surface
Training production ML models — dataset too small and domain-specific
Generalising findings without accounting for specific function characteristics
Distribution and Maintenance
Where is the dataset available? https://github.com/stevenasuarez/imperial-capstone-project

Query records stored in:

src/data_loader.py — all round coordinates as Python dictionaries
results/queries_roundN.txt — portal-format strings per round
Terms of use: Created for educational purposes as part of the Imperial College London / Emeritus PCMLAI programme. May be freely used for academic and educational purposes with attribution. Oracle outputs are the intellectual property of Imperial College London / Emeritus and are reproduced solely to document the author's own submissions.

Who maintains it? Steven Suarez.