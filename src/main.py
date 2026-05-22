# src/main.py
# BBO Capstone - Full Query Strategy (Rounds 1-8)
# Imperial College / Emeritus PCMLAI Capstone
#
# USAGE:
#   python main.py --round 2   → analyse R1→R2 changes
#   python main.py --round 3   → generate Round 3 (SVM-guided)
#   python main.py --round 4   → generate Round 4 (numpy NN surrogate)
#   python main.py --round 5   → generate Round 5 (PyTorch surrogate + autograd)
#   python main.py --round 6   → generate Round 6 (PyTorch + dimension-aware pooling)
#   python main.py --round 7   → generate Round 7 (PyTorch + hyperparameter grid search)
#   python main.py --round 8   → generate Round 8 (PyTorch + attention-weighted gradient)
#   python main.py             → runs all rounds in sequence
#
# REQUIREMENTS:
#   pip install numpy scikit-learn torch

import argparse
import numpy as np
import os
from data_loader import round1, round2, round3, round4, round5, round6, round7
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
# ═══════════════════════════════════════════════════════════════

def run_round5():
    print("\n" + "=" * 60)
    print("ROUND 5 PROPOSED QUERIES (PyTorch surrogate + autograd)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        class PyTorchSurrogate(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 8), nn.ReLU(),
                    nn.Linear(8, 1), nn.Sigmoid()
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
            X_t = torch.tensor(np.array([p1,p2,p3,p4], dtype=np.float32))
            y_t = torch.tensor(np.array([[0.0],[0.33],[0.67],[1.0]], dtype=np.float32))
            torch.manual_seed(42)
            model = PyTorchSurrogate(input_dim=dim)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            for _ in range(5000):
                opt.zero_grad(); loss_fn(model(X_t), y_t).backward(); opt.step()
            x_query = torch.tensor(p4, dtype=torch.float32, requires_grad=True)
            model(x_query.unsqueeze(0)).backward()
            grad = x_query.grad.numpy()
            grad_norm = grad / (np.linalg.norm(grad) + 1e-8)
            avg_step = (np.linalg.norm(p2-p1)+np.linalg.norm(p3-p2)+np.linalg.norm(p4-p3))/3
            p5 = np.clip(p4 + avg_step * 0.35 * grad_norm, 0.0, 1.0)
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
        print("\n  PyTorch not found. Install with: pip install torch")


# ═══════════════════════════════════════════════════════════════
# ROUND 6 — PyTorch surrogate + dimension-aware pooling
# ═══════════════════════════════════════════════════════════════

def run_round6():
    print("\n" + "=" * 60)
    print("ROUND 6 PROPOSED QUERIES (PyTorch + dimension-aware pooling)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        class PyTorchSurrogate(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 8), nn.ReLU(),
                    nn.Linear(8, 1), nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)

        all_rounds   = [round1, round2, round3, round4, round5]
        label_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = {}

        for key in round1:
            points = [np.array(r[key], dtype=np.float32) for r in all_rounds]
            dim    = len(points[0])
            activity = np.zeros(dim)
            for i in range(1, len(points)):
                activity += np.abs(points[i] - points[i-1])
            activity = activity / (activity.max() + 1e-8)
            activity = 0.2 + 0.8 * activity
            X_t = torch.tensor(np.array(points, dtype=np.float32))
            y_t = torch.tensor(np.array([[v] for v in label_values], dtype=np.float32))
            torch.manual_seed(42)
            model = PyTorchSurrogate(input_dim=dim)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            for _ in range(5000):
                opt.zero_grad(); loss_fn(model(X_t), y_t).backward(); opt.step()
            p5 = points[-1].copy()
            for d in range(dim):
                if p5[d] <= 0.01: p5[d] = 0.05
                elif p5[d] >= 0.99: p5[d] = 0.95
            x_query = torch.tensor(p5, dtype=torch.float32, requires_grad=True)
            model(x_query.unsqueeze(0)).backward()
            grad = x_query.grad.numpy()
            grad_norm = (grad * activity) / (np.linalg.norm(grad * activity) + 1e-8)
            recent_steps = [np.linalg.norm(points[i]-points[i-1]) for i in range(2, len(points))]
            p6 = np.clip(p5 + np.mean(recent_steps) * 0.3 * grad_norm, 0.0, 1.0)
            result[key] = [round(float(x), 6) for x in p6]

        lines = []
        for key in result:
            portal_format = "-".join(f"{x:.6f}" for x in result[key])
            print(f"\n{key}:")
            print(f"  Round 5       : {round5[key]}")
            print(f"  Round 6       : {result[key]}")
            print(f"  Portal format : {portal_format}")
            lines.append(f"{key}: {portal_format}")
        _save(lines, "queries_round6.txt")
    except ImportError:
        print("\n  PyTorch not found. Install with: pip install torch")


# ═══════════════════════════════════════════════════════════════
# ROUND 7 — PyTorch surrogate + hyperparameter grid search
# ═══════════════════════════════════════════════════════════════

def run_round7():
    print("\n" + "=" * 60)
    print("ROUND 7 PROPOSED QUERIES (PyTorch + hyperparameter grid search)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        def build_model(input_dim, hidden_size):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 1), nn.Sigmoid()
            )

        def train_model(model, X_t, y_t, lr, epochs=2000):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            for _ in range(epochs):
                opt.zero_grad(); loss_fn(model(X_t), y_t).backward(); opt.step()
            return model

        def loo_cv_loss(X_np, y_np, lr, hidden_size, input_dim):
            n = len(X_np); total = 0.0
            for i in range(n):
                X_t = torch.tensor(np.delete(X_np, i, axis=0), dtype=torch.float32)
                y_t = torch.tensor(np.delete(y_np, i, axis=0), dtype=torch.float32)
                torch.manual_seed(42)
                model = train_model(build_model(input_dim, hidden_size), X_t, y_t, lr)
                with torch.no_grad():
                    pred = model(torch.tensor(X_np[i:i+1], dtype=torch.float32))
                    total += float((pred - torch.tensor(y_np[i:i+1], dtype=torch.float32))**2)
            return total / n

        lr_grid = [0.001, 0.005, 0.01, 0.05]
        hidden_grid = [16, 32]
        step_scale_grid = [0.2, 0.3, 0.4]
        all_rounds = [round1, round2, round3, round4, round5, round6]
        label_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        result = {}

        for key in round1:
            points = [np.array(r[key], dtype=np.float32) for r in all_rounds]
            dim = len(points[0])
            X_np = np.array(points, dtype=np.float32)
            y_np = np.array([[v] for v in label_values], dtype=np.float32)
            best_loss = float('inf'); best_lr = 0.01; best_hidden = 32
            print(f"\n  {key}: running grid search...", end="", flush=True)
            for lr in lr_grid:
                for hs in hidden_grid:
                    loss = loo_cv_loss(X_np, y_np, lr, hs, dim)
                    if loss < best_loss:
                        best_loss = loss; best_lr = lr; best_hidden = hs
            print(f" best lr={best_lr}, hidden={best_hidden}, cv_loss={best_loss:.6f}")
            X_t = torch.tensor(X_np); y_t = torch.tensor(y_np)
            torch.manual_seed(42)
            model = train_model(build_model(dim, best_hidden), X_t, y_t, best_lr, epochs=5000)
            activity = np.zeros(dim)
            for i in range(1, len(points)):
                activity += np.abs(points[i] - points[i-1])
            activity = 0.2 + 0.8 * activity / (activity.max() + 1e-8)
            p6 = points[-1].copy()
            for d in range(dim):
                if p6[d] <= 0.01: p6[d] = 0.05
                elif p6[d] >= 0.99: p6[d] = 0.95
            x_query = torch.tensor(p6, dtype=torch.float32, requires_grad=True)
            model(x_query.unsqueeze(0)).backward()
            grad = x_query.grad.numpy()
            grad_norm = (grad * activity) / (np.linalg.norm(grad * activity) + 1e-8)
            recent_steps = [np.linalg.norm(points[i]-points[i-1]) for i in range(3, len(points))]
            base_step = np.mean(recent_steps)
            best_scale = 0.3
            for scale in step_scale_grid:
                if np.all(np.clip(p6 + base_step*scale*grad_norm, 0, 1) == p6 + base_step*scale*grad_norm):
                    best_scale = scale; break
            p7 = np.clip(p6 + base_step * best_scale * grad_norm, 0.0, 1.0)
            result[key] = [round(float(x), 6) for x in p7]

        lines = []
        for key in result:
            portal_format = "-".join(f"{x:.6f}" for x in result[key])
            print(f"\n{key}:")
            print(f"  Round 6       : {round6[key]}")
            print(f"  Round 7       : {result[key]}")
            print(f"  Portal format : {portal_format}")
            lines.append(f"{key}: {portal_format}")
        _save(lines, "queries_round7.txt")
    except ImportError:
        print("\n  PyTorch not found. Install with: pip install torch")


# ═══════════════════════════════════════════════════════════════
# ROUND 8 — PyTorch surrogate + attention-weighted gradient
#
# Module focus: transformers, multi-head attention, tokenisation
#
# Key concept: in transformers, attention scores determine how
# much each position attends to every other position. Here we
# apply the same idea across the TIME dimension of our search:
#
#   - Treat each round as a "token" in a sequence
#   - Compute attention scores between the current query point
#     (Round 7) and all previous rounds using scaled dot-product
#     attention: softmax(QK^T / sqrt(d)) * V
#   - Rounds more similar to the current point get higher
#     attention weights — they are more "relevant context"
#   - Use attention weights to compute a weighted centroid of
#     historical movements, forming an attention-guided
#     direction for Round 8
#   - Combine with the surrogate gradient for the final step
#
# This replaces the fixed activity mask from Rounds 6-7 with
# a dynamic, query-dependent attention mechanism.
# ═══════════════════════════════════════════════════════════════

def run_round8():
    print("\n" + "=" * 60)
    print("ROUND 8 PROPOSED QUERIES (PyTorch + attention-weighted gradient)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        def build_model(input_dim, hidden_size=32):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 1), nn.Sigmoid()
            )

        def train_model(model, X_t, y_t, lr=0.01, epochs=5000):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            for _ in range(epochs):
                opt.zero_grad(); loss_fn(model(X_t), y_t).backward(); opt.step()
            return model

        def scaled_dot_product_attention(query, keys):
            """
            Transformer-style scaled dot-product attention.
            query : (dim,)     — the current Round 7 point
            keys  : (n, dim)   — all previous round points
            Returns attention weights (n,) summing to 1.
            Higher weight = more similar to current query = more relevant.

            attention(Q, K) = softmax(QK^T / sqrt(d))
            """
            d = query.shape[0]
            # Dot product between query and each key (round)
            scores = keys @ query / np.sqrt(d)
            # Softmax to get normalised attention weights
            scores = scores - scores.max()  # numerical stability
            weights = np.exp(scores) / np.sum(np.exp(scores))
            return weights

        all_rounds   = [round1, round2, round3, round4, round5, round6, round7]
        label_values = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]

        result = {}

        for key in round1:
            points = [np.array(r[key], dtype=np.float32) for r in all_rounds]
            dim    = len(points[0])

            # ── Train surrogate on all 7 rounds ──
            X_np = np.array(points, dtype=np.float32)
            y_np = np.array([[v] for v in label_values], dtype=np.float32)
            X_t  = torch.tensor(X_np)
            y_t  = torch.tensor(y_np)

            torch.manual_seed(42)
            model = build_model(dim, hidden_size=32)
            model = train_model(model, X_t, y_t, lr=0.01, epochs=5000)

            # ── Compute surrogate gradient at Round 7 point ──
            p7 = points[-1].copy()
            for d in range(dim):
                if p7[d] <= 0.01: p7[d] = 0.05
                elif p7[d] >= 0.99: p7[d] = 0.95

            x_query = torch.tensor(p7, dtype=torch.float32, requires_grad=True)
            model(x_query.unsqueeze(0)).backward()
            surrogate_grad = x_query.grad.numpy()

            # ── Compute attention weights over previous rounds ──
            # Query = current point (Round 7)
            # Keys  = all previous round points
            # Rounds more similar to Round 7 get higher attention
            query = p7                          # (dim,)
            keys  = np.array(points[:-1])       # (n_prev, dim)
            attn_weights = scaled_dot_product_attention(query, keys)

            # ── Attention-weighted movement direction ──
            # For each previous round, compute the movement vector
            # to the next round. Weight each by attention score.
            # This asks: "what movements from similar past states
            # were most useful?" — the transformer's key insight.
            movements = []
            for i in range(1, len(points) - 1):
                move = points[i] - points[i-1]
                movements.append(move)
            movements = np.array(movements)             # (n_prev-1, dim)
            attn_weights_moves = attn_weights[1:]       # align with movements
            attn_weights_moves = attn_weights_moves / (attn_weights_moves.sum() + 1e-8)
            attention_direction = attn_weights_moves @ movements  # (dim,)

            # ── Combine surrogate gradient + attention direction ──
            # 60% surrogate gradient (exploitation)
            # 40% attention-weighted historical direction (context)
            surrogate_norm  = surrogate_grad / (np.linalg.norm(surrogate_grad) + 1e-8)
            attention_norm  = attention_direction / (np.linalg.norm(attention_direction) + 1e-8)
            combined        = 0.6 * surrogate_norm + 0.4 * attention_norm
            combined_norm   = combined / (np.linalg.norm(combined) + 1e-8)

            # ── Step size: mean of last 3 round movements * 0.25 ──
            recent_steps = [np.linalg.norm(points[i] - points[i-1])
                            for i in range(4, len(points))]
            step_size = np.mean(recent_steps) * 0.25

            p8 = np.clip(p7 + step_size * combined_norm, 0.0, 1.0)
            result[key] = [round(float(x), 6) for x in p8]

        lines = []
        for key in result:
            portal_format = "-".join(f"{x:.6f}" for x in result[key])
            print(f"\n{key}:")
            print(f"  Round 7       : {round7[key]}")
            print(f"  Round 8       : {result[key]}")
            print(f"  Portal format : {portal_format}")
            lines.append(f"{key}: {portal_format}")
        _save(lines, "queries_round8.txt")

    except ImportError:
        print("\n  PyTorch not found. Install with: pip install torch")
        print("  Then re-run: python main.py --round 8")


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
        "--round", type=int, choices=[2, 3, 4, 5, 6, 7, 8],
        help="Which round to run (2-8). Omit to run all."
    )
    args = parser.parse_args()

    if args.round == 2:   run_round2()
    elif args.round == 3: run_round3()
    elif args.round == 4: run_round4()
    elif args.round == 5: run_round5()
    elif args.round == 6: run_round6()
    elif args.round == 7: run_round7()
    elif args.round == 8: run_round8()
    else:
        run_round2(); run_round3(); run_round4()
        run_round5(); run_round6(); run_round7()
        run_round8()