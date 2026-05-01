# src/main.py
# BBO Capstone - Full Query Strategy (Rounds 1-5)
# Imperial College / Emeritus PCMLAI Capstone
#
# USAGE:
#   python main.py --round 2   → analyse R1→R2 changes
#   python main.py --round 3   → generate Round 3 (SVM-guided)
#   python main.py --round 4   → generate Round 4 (numpy NN surrogate)
#   python main.py --round 5   → generate Round 5 (PyTorch surrogate + autograd)
#   python main.py             → runs all rounds in sequence
#
# REQUIREMENTS:
#   pip install numpy scikit-learn torch

import argparse
import numpy as np
import os
from data_loader import round1, round2, round3, round4
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════
# ROUND 2 — Analyse changes from Round 1 → Round 2
# ═══════════════════════════════════════════════════════════════

def run_round2():
    print("\n" + "=" * 60)
    print("ROUND 1 → ROUND 2 ANALYSIS")
    print("=" * 60)
    for key in round1:
        changes = [round(round2[key][i] - round1[key][i], 6)
                   for i in range(len(round1[key]))]
        print(f"\n{key}:")
        print(f"  Round 1 : {round1[key]}")
        print(f"  Round 2 : {round2[key]}")
        print(f"  Change  : {changes}")


# ═══════════════════════════════════════════════════════════════
# ROUND 3 — SVM-guided search
#
# Strategy:
#   - Label Round 1 as 0 (baseline) and Round 2 as 1 (improved)
#   - Train a linear SVM to find the decision boundary
#   - Extract the SVM weight vector as a direction of improvement
#   - Step from Round 2 along that direction to propose Round 3
# ═══════════════════════════════════════════════════════════════

def generate_round3_svm(r1, r2, step_scale=0.5):
    result = {}
    for key in r1:
        p1 = np.array(r1[key])
        p2 = np.array(r2[key])
        X = np.array([p1, p2])
        y = np.array([0, 1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_scaled, y)
        w = svm.coef_[0] / scaler.scale_
        w_norm = w / (np.linalg.norm(w) + 1e-8)
        step_size = np.linalg.norm(p2 - p1) * step_scale
        p3 = np.clip(p2 + step_size * w_norm, 0.0, 1.0)
        result[key] = [round(float(x), 6) for x in p3]
    return result

def run_round3():
    print("\n" + "=" * 60)
    print("ROUND 3 PROPOSED QUERIES (SVM-guided)")
    print("=" * 60)
    result = generate_round3_svm(round1, round2, step_scale=0.5)
    lines = []
    for key in result:
        portal_format = "-".join(f"{x:.6f}" for x in result[key])
        print(f"\n{key}:")
        print(f"  Round 2       : {round2[key]}")
        print(f"  Round 3       : {result[key]}")
        print(f"  Portal format : {portal_format}")
        lines.append(f"{key}: {portal_format}")
    _save(lines, "queries_round3.txt")


# ═══════════════════════════════════════════════════════════════
# ROUND 4 — Numpy NN surrogate + manual backpropagation
#
# Strategy:
#   - Assign synthetic labels: R1=0.0, R2=0.5, R3=1.0
#   - Train a small NN from scratch using numpy
#   - Use manual backpropagation to compute d(output)/d(input)
#   - Step from Round 3 along that gradient to propose Round 4
# ═══════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


class SurrogateNN:
    """
    Fully-connected NN built from scratch with numpy only.
    Architecture: Input(dim) -> Hidden(16, ReLU) -> Hidden(8, ReLU) -> Output(1, Sigmoid)
    Trained with MSE loss and standard gradient descent.
    input_gradient() uses manual backpropagation to compute d(output)/d(input).
    """
    def __init__(self, input_dim, lr=0.05, epochs=3000):
        self.lr = lr
        self.epochs = epochs
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, 16) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, 16))
        self.W2 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
        self.b2 = np.zeros((1, 8))
        self.W3 = np.random.randn(8, 1) * np.sqrt(2.0 / 8)
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1;   self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2;  self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3;  self.out = sigmoid(self.z3)
        return self.out

    def backward(self, X, y):
        n = X.shape[0]
        d3 = (self.out - y) * sigmoid_deriv(self.z3)
        dW3 = self.a2.T @ d3 / n;  db3 = d3.mean(axis=0, keepdims=True)
        d2 = (d3 @ self.W3.T) * relu_deriv(self.z2)
        dW2 = self.a1.T @ d2 / n;  db2 = d2.mean(axis=0, keepdims=True)
        d1 = (d2 @ self.W2.T) * relu_deriv(self.z1)
        dW1 = X.T @ d1 / n;  db1 = d1.mean(axis=0, keepdims=True)
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def train(self, X, y):
        for _ in range(self.epochs):
            self.forward(X)
            self.backward(X, y)

    def input_gradient(self, x):
        self.forward(x.reshape(1, -1))
        d3 = sigmoid_deriv(self.z3)
        d2 = (d3 @ self.W3.T) * relu_deriv(self.z2)
        d1 = (d2 @ self.W2.T) * relu_deriv(self.z1)
        return (d1 @ self.W1.T).flatten()


def generate_round4_nn(r1, r2, r3, step_scale=0.4):
    result = {}
    for key in r1:
        p1 = np.array(r1[key])
        p2 = np.array(r2[key])
        p3 = np.array(r3[key])
        X = np.array([p1, p2, p3])
        y = np.array([[0.0], [0.5], [1.0]])
        nn = SurrogateNN(input_dim=len(p1), lr=0.05, epochs=3000)
        nn.train(X, y)
        grad = nn.input_gradient(p3)
        grad_norm = grad / (np.linalg.norm(grad) + 1e-8)
        avg_step = (np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)) / 2
        step_size = avg_step * step_scale
        p4 = np.clip(p3 + step_size * grad_norm, 0.0, 1.0)
        result[key] = [round(float(x), 6) for x in p4]
    return result

def run_round4():
    print("\n" + "=" * 60)
    print("ROUND 4 PROPOSED QUERIES (Numpy NN surrogate + manual backprop)")
    print("=" * 60)
    result = generate_round4_nn(round1, round2, round3, step_scale=0.4)
    lines = []
    for key in result:
        portal_format = "-".join(f"{x:.6f}" for x in result[key])
        print(f"\n{key}:")
        print(f"  Round 3       : {round3[key]}")
        print(f"  Round 4       : {result[key]}")
        print(f"  Portal format : {portal_format}")
        lines.append(f"{key}: {portal_format}")
    _save(lines, "queries_round4.txt")


# ═══════════════════════════════════════════════════════════════
# ROUND 5 — PyTorch surrogate + autograd gradient search
#
# Strategy:
#   - Upgrade the surrogate to PyTorch, which handles all
#     gradient computation automatically via autograd
#   - Train on all 4 rounds with synthetic progressive labels:
#     R1=0.0, R2=0.33, R3=0.67, R4=1.0
#   - After training, use torch.autograd to compute
#     d(output)/d(input) at Round 4 — no manual backprop needed
#   - Deeper architecture (32→16→8→1) to better capture
#     non-linearities as the dataset grows to 4 points
#   - Adam optimiser for faster, more stable convergence
#     vs plain gradient descent used in Round 4
# ═══════════════════════════════════════════════════════════════

def run_round5():
    print("\n" + "=" * 60)
    print("ROUND 5 PROPOSED QUERIES (PyTorch surrogate + autograd)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        class PyTorchSurrogate(nn.Module):
            """
            Deeper surrogate network built with PyTorch.
            Architecture: Input(dim) -> 32 -> 16 -> 8 -> Output(1)
            Activations: ReLU throughout, Sigmoid on output.

            Using PyTorch means:
            - No manual backpropagation code needed
            - torch.autograd handles all gradient computation
            - Adam optimiser gives adaptive learning rates
            - Easy to scale architecture without rewriting math
            """
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x)

        result = {}
        for key in round1:
            p1 = np.array(round1[key], dtype=np.float32)
            p2 = np.array(round2[key], dtype=np.float32)
            p3 = np.array(round3[key], dtype=np.float32)
            p4 = np.array(round4[key], dtype=np.float32)
            dim = len(p1)

            # Training data: 4 rounds, progressive synthetic labels
            X_np = np.array([p1, p2, p3, p4], dtype=np.float32)
            y_np = np.array([[0.0], [0.33], [0.67], [1.0]], dtype=np.float32)

            X_t = torch.tensor(X_np)
            y_t = torch.tensor(y_np)

            # Build and train PyTorch model
            torch.manual_seed(42)
            model = PyTorchSurrogate(input_dim=dim)
            optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            for epoch in range(5000):
                optimiser.zero_grad()
                pred = model(X_t)
                loss = loss_fn(pred, y_t)
                loss.backward()   # PyTorch autograd — no manual math needed
                optimiser.step()

            # Compute input gradient at Round 4 point using autograd
            # requires_grad=True tells PyTorch to track this tensor
            x_query = torch.tensor(p4, dtype=torch.float32,
                                   requires_grad=True)
            output = model(x_query.unsqueeze(0))
            output.backward()  # autograd computes d(output)/d(x_query)
            grad = x_query.grad.numpy()  # gradient of output w.r.t. each input dim

            # Step along gradient from Round 4 to propose Round 5
            grad_norm = grad / (np.linalg.norm(grad) + 1e-8)
            avg_step = (np.linalg.norm(p2 - p1) +
                        np.linalg.norm(p3 - p2) +
                        np.linalg.norm(p4 - p3)) / 3
            step_size = avg_step * 0.35

            p5 = np.clip(p4 + step_size * grad_norm, 0.0, 1.0)
            result[key] = [round(float(x), 6) for x in p5]

        lines = []
        for key in result:
            portal_format = "-".join(f"{x:.6f}" for x in result[key])
            print(f"\n{key}:")
            print(f"  Round 4       : {round4[key]}")
            print(f"  Round 5       : {result[key]}")
            print(f"  Portal format : {portal_format}")
            lines.append(f"{key}: {portal_format}")
        _save(lines, "queries_round5.txt")

    except ImportError:
        print("\n  PyTorch not found. Install it with: pip install torch")
        print("  Then re-run: python main.py --round 5")


# ═══════════════════════════════════════════════════════════════
# HELPER — Save results to ../results/
# ═══════════════════════════════════════════════════════════════

def _save(lines, filename):
    for path in [f"../results/{filename}", f"results/{filename}"]:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write("\n".join(lines))
            print(f"\n  Saved to {path}")
            return
        except Exception:
            continue


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBO Capstone - Query Generator")
    parser.add_argument(
        "--round", type=int, choices=[2, 3, 4, 5],
        help="Which round to run (2, 3, 4 or 5). Omit to run all."
    )
    args = parser.parse_args()

    if args.round == 2:
        run_round2()
    elif args.round == 3:
        run_round3()
    elif args.round == 4:
        run_round4()
    elif args.round == 5:
        run_round5()
    else:
        run_round2()
        run_round3()
        run_round4()
        run_round5()