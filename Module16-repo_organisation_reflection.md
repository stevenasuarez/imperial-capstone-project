## GitHub Repository Organisation – BBO Capstone

---

### Repository Structure

The repository is organised into four main directories that reflect the natural separation between code, data, outputs and analysis:

```
Capstone/
├── src/
│   ├── main.py          # All round strategies (2-5) in one unified script
│   └── data_loader.py   # All query data across rounds 1-5
├── results/
│   ├── queries_round1.txt
│   ├── queries_round2.txt
│   ├── queries_round3.txt
│   ├── queries_round4.txt
│   └── queries_round5.txt
├── notebooks/
│   └── stage2_analysis.ipynb
├── requirements.txt
└── README.md
```

The most deliberate structural decision was keeping all round strategies inside a single `main.py` rather than creating separate files per round. This keeps the evolution of the approach visible in one place — you can read the file top to bottom and see the progression from SVM-guided search (Round 3) through numpy backpropagation (Round 4) to PyTorch autograd (Round 5). Each round is a self-contained function callable via `python main.py --round N`, which makes individual rounds reproducible without running the full pipeline.

**Planned improvements:**
- Add a `reflections/` folder to store the written round-by-round strategy documents alongside the code, making the reasoning traceable directly from the repo
- Add inline comments to `data_loader.py` noting which rounds produced stuck queries (f4 across all rounds, f6 in Round 5) and why, so the data tells its own story
- Add a `.gitignore` to exclude `__pycache__` and `.venv` directories that don't belong in version control

---

### Coding Libraries and Packages

Three libraries are central to the project:

**numpy** — used throughout for array operations, vector arithmetic and the manual backpropagation implementation in Round 4. It was the right starting point because it forced explicit implementation of the chain rule, making the mechanics of gradient computation transparent before abstracting them away.

**scikit-learn** — used in Round 3 for the SVM surrogate. `SVC` with a linear kernel and `StandardScaler` for feature normalisation provided a clean, interpretable directional signal from just two labelled points per function. scikit-learn's consistency and well-documented API made it easy to extract the weight vector as a geometric direction of improvement.

**PyTorch** — introduced in Round 5 as the primary surrogate framework. The switch from numpy to PyTorch was motivated by three things: automatic differentiation via `autograd` eliminates manual backpropagation code, the `Adam` optimiser provides adaptive learning rates without tuning a single global step size, and the `nn.Sequential` API makes architectural changes (adding layers, swapping activations) a one-line operation. PyTorch's dynamic computation graph also aligns with the iterative, experimental nature of the capstone — the model can be restructured between rounds without rewriting the training loop.

TensorFlow was considered but not adopted. For a research-oriented, rapidly evolving project like this one, PyTorch's eager execution and Pythonic debugging experience were more appropriate than TensorFlow's production-focused static graph design.

---

### Documentation

The `README.md` currently covers all four required sections: project overview (including career relevance as a QA Automation Engineer), inputs and outputs with format examples and a dimensionality table, challenge objectives stating the maximisation goal and query constraints, and a technical approach section that documents the evolution across all five rounds as a living record.

The README was deliberately written as a narrative document rather than a bullet-point summary, so that a reader unfamiliar with the project can understand not just what was done but why each strategy change was made.

**Updates needed to reflect current state:**
- Add Round 5 to the technical approach section, describing the PyTorch upgrade and the autograd gradient search
- Add a known limitations note documenting the f4 and f6 stagnation issue and the synthetic labelling assumption (since oracle scores have not been returned by the portal)
- Update the requirements section to include `torch==2.11.0` alongside numpy and scikit-learn
- Add a usage section showing the `python main.py --round N` command pattern, so the repo is immediately runnable by anyone who clones it

The goal is for the repository to function as a transparent record of an evolving optimisation strategy — not just a code dump, but a document of decisions made under uncertainty, which is the core skill the capstone develops.
