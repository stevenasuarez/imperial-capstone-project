# src/main.py
# BBO Capstone - Full Query Strategy (Rounds 1-13)
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
#   python main.py --round 9   → generate Round 9 (real scores + score-guided strategy)
#   python main.py --round 10  → generate Round 10 (real-score surrogate + interpretable)
#   python main.py --round 11  → generate Round 11 (clustering-based strategy)
#   python main.py --round 12  → generate Round 12 (PCA-guided search)
#   python main.py --round 13  → generate Round 13 (FINAL — pure exploitation)
#   python main.py             → runs all rounds in sequence
#
# REQUIREMENTS:
#   pip install numpy scikit-learn torch

import argparse
import numpy as np
import os
from data_loader import round1, round2, round3, round4, round5, round6, round7, round8, round9, round10, round11, round12
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
# ═══════════════════════════════════════════════════════════════

def run_round8():
    print("\n" + "=" * 60)
    print("ROUND 8 PROPOSED QUERIES (PyTorch + attention-weighted gradient)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

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
            d = query.shape[0]
            scores = keys @ query / np.sqrt(d)
            scores = scores - scores.max()
            weights = np.exp(scores) / np.sum(np.exp(scores))
            return weights

        all_rounds   = [round1, round2, round3, round4, round5, round6, round7]
        label_values = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]
        result = {}

        for key in round1:
            points = [np.array(r[key], dtype=np.float32) for r in all_rounds]
            dim    = len(points[0])
            X_t = torch.tensor(np.array(points, dtype=np.float32))
            y_t = torch.tensor(np.array([[v] for v in label_values], dtype=np.float32))
            torch.manual_seed(42)
            model = build_model(dim)
            model = train_model(model, X_t, y_t)
            p7 = points[-1].copy()
            for d in range(dim):
                if p7[d] <= 0.01: p7[d] = 0.05
                elif p7[d] >= 0.99: p7[d] = 0.95
            x_query = torch.tensor(p7, dtype=torch.float32, requires_grad=True)
            model(x_query.unsqueeze(0)).backward()
            surrogate_grad = x_query.grad.numpy()
            query = p7
            keys  = np.array(points[:-1])
            attn_weights = scaled_dot_product_attention(query, keys)
            movements = np.array([points[i] - points[i-1] for i in range(1, len(points)-1)])
            attn_w = attn_weights[1:] / (attn_weights[1:].sum() + 1e-8)
            attention_direction = attn_w @ movements
            surrogate_norm = surrogate_grad / (np.linalg.norm(surrogate_grad) + 1e-8)
            attention_norm = attention_direction / (np.linalg.norm(attention_direction) + 1e-8)
            combined = 0.6 * surrogate_norm + 0.4 * attention_norm
            combined_norm = combined / (np.linalg.norm(combined) + 1e-8)
            recent_steps = [np.linalg.norm(points[i]-points[i-1]) for i in range(4, len(points))]
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


# ═══════════════════════════════════════════════════════════════
# ROUND 9 — Real score-guided strategy
#
# GAME CHANGER: we now have real oracle scores for all rounds
# 3-8. This completely replaces synthetic labelling.
#
# Full score history per function:
#   W3      W4       W5       W6       W7       W8
# f1: ~0     ~0       ~0       ~0       ~0       ~0    → flat, reset
# f2: 0.725  0.435    0.541    0.531    0.582    0.537 → noisy, near best at W3
# f3: -0.127 -0.159   -0.179   -0.170   -0.162   -0.159 → wrong dir, reset
# f4: -4.378 -4.378   -6.056   -6.254   -6.379   -6.494 → getting worse, hard reset
# f5: 445.9  1320.3   1623.0   658.4    542.7    520.5 → PEAKED W5, move back there
# f6: -0.552 -0.611   -0.613   -0.576   -0.581   -0.580 → consistently negative
# f7: 1.187  1.188    1.186    1.185    1.185    1.184 → very stable, tiny decline
# f8: 8.070  8.072    8.071    8.071    8.060    8.059 → very stable, tiny decline
#
# Strategy:
#   f1 → reset to [0.5, 0.5] (completely flat everywhere we've tried)
#   f2 → move back toward Week 3 coordinates (best score was there)
#   f3 → reset toward centre (consistently negative)
#   f4 → hard reset toward centre (getting dramatically worse)
#   f5 → move back toward Week 5 coordinates (best score was 1623 there)
#   f6 → reset toward centre (consistently negative)
#   f7 → tiny exploitation step (stable, very slightly declining)
#   f8 → tiny exploitation step (stable, very slightly declining)
# ═══════════════════════════════════════════════════════════════

def run_round9():
    print("\n" + "=" * 60)
    print("ROUND 9 PROPOSED QUERIES (Real score-guided strategy)")
    print("=" * 60)

    # All real scores indexed by week (3-8) and function
    all_scores = {
        "f1": [6.36e-25, 2.33e-25, 1.19e-26, 2.64e-27, 1.42e-27, 8.76e-28],
        "f2": [0.7247,   0.4346,   0.5414,   0.5306,   0.5819,   0.5375],
        "f3": [-0.1268,  -0.1590,  -0.1790,  -0.1701,  -0.1620,  -0.1587],
        "f4": [-4.378,   -4.378,   -6.056,   -6.254,   -6.379,   -6.494],
        "f5": [445.89,   1320.26,  1623.03,  658.43,   542.69,   520.50],
        "f6": [-0.5522,  -0.6107,  -0.6126,  -0.5762,  -0.5812,  -0.5800],
        "f7": [1.1865,   1.1880,   1.1862,   1.1852,   1.1848,   1.1843],
        "f8": [8.0703,   8.0724,   8.0712,   8.0712,   8.0598,   8.0593],
    }

    # Rounds corresponding to weeks 3-8
    scored_rounds = [round3, round4, round5, round6, round7, round8]

    result = {}

    for key in round1:
        scores = all_scores[key]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        current_point = np.array(round8[key], dtype=np.float32)
        best_point = np.array(scored_rounds[best_idx][key], dtype=np.float32)
        dim = len(current_point)

        print(f"\n  {key}: best_score={best_score:.4f} at week {best_idx+3}, "
              f"current={scores[-1]:.4f}")

        if key == "f1":
            # Completely flat everywhere — reset to centre
            p9 = np.full(dim, 0.5, dtype=np.float32)
            print(f"    → RESET to centre")

        elif key == "f2":
            # Best was Week 3 (0.7247). Move back toward that point
            direction = best_point - current_point
            dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
            step_size = np.linalg.norm(direction) * 0.6
            p9 = np.clip(current_point + step_size * dir_norm, 0.0, 1.0)
            print(f"    → RETURN toward best point (W{best_idx+3})")

        elif key == "f3":
            # Consistently negative — reset toward centre
            centre = np.full(dim, 0.5, dtype=np.float32)
            direction = centre - current_point
            p9 = np.clip(current_point + direction * 0.7, 0.0, 1.0)
            print(f"    → RESET toward centre")

        elif key == "f4":
            # Getting dramatically worse — hard reset to centre
            p9 = np.full(dim, 0.5, dtype=np.float32)
            print(f"    → HARD RESET to centre")

        elif key == "f5":
            # Peaked at Week 5 (1623) — move back toward those coordinates
            direction = best_point - current_point
            dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
            step_size = np.linalg.norm(direction) * 0.7
            p9 = np.clip(current_point + step_size * dir_norm, 0.0, 1.0)
            print(f"    → RETURN toward best point (W{best_idx+3}, score={best_score:.1f})")

        elif key == "f6":
            # Consistently negative — reset toward centre
            centre = np.full(dim, 0.5, dtype=np.float32)
            direction = centre - current_point
            p9 = np.clip(current_point + direction * 0.7, 0.0, 1.0)
            print(f"    → RESET toward centre")

        elif key in ["f7", "f8"]:
            # Stable and positive — tiny exploitation step in same direction
            last_move = np.array(round8[key]) - np.array(round7[key])
            move_norm = last_move / (np.linalg.norm(last_move) + 1e-8)
            step_size = np.linalg.norm(last_move) * 0.5
            p9 = np.clip(current_point + step_size * move_norm, 0.0, 1.0)
            print(f"    → EXPLOIT: continue small step")

        else:
            p9 = current_point

        result[key] = [round(float(x), 6) for x in p9]

    lines = []
    for key in result:
        portal_format = "-".join(f"{x:.6f}" for x in result[key])
        print(f"\n{key}:")
        print(f"  Round 8       : {round8[key]}")
        print(f"  Round 9       : {result[key]}")
        print(f"  Portal format : {portal_format}")
        lines.append(f"{key}: {portal_format}")
    _save(lines, "queries_round9.txt")


# ROUND 10 function — paste this into main.py

# ═══════════════════════════════════════════════════════════════
# ROUND 10 — Real-score surrogate + fully interpretable decisions
#
# Module focus: transparency and interpretability
#
# This is the most principled round yet. We now have real oracle
# scores for 7 consecutive rounds (W3-W9). This enables:
#
#   1. REAL LABELS: surrogate trained on actual oracle outputs
#      (normalised to [0,1]) instead of synthetic progressions
#
#   2. INTERPRETABLE DECISIONS: for each function, the code
#      prints the exact reasoning — which historical point was
#      best, what direction is being taken, and why
#
#   3. SCORE-BASED ROUTING: per-function strategy chosen
#      explicitly based on score trend analysis:
#      - improving  → continue in same direction
#      - declining  → return toward historical best
#      - recovering → keep recovering (reset worked last round)
#      - negative   → try new region or reverse
#
# Per-function score analysis (W3→W9):
#   f1: ~0 everywhere → explore new region from centre
#   f2: 0.72→0.57 noisy, W9=0.573 improving → small step
#   f3: negative, W9=-0.058 (best ever) → keep recovering
#   f4: -4.38→-3.99 improving from reset → continue toward centre
#   f5: 446→1623→520→1189, W9=1189 → keep going this direction
#   f6: W9=-0.764 worse after reset → return to pre-reset region
#   f7: stable 1.184-1.188 → tiny exploitation step
#   f8: stable 8.059-8.072 → tiny exploitation step
# ═══════════════════════════════════════════════════════════════

def run_round10():
    print("\n" + "=" * 60)
    print("ROUND 10 PROPOSED QUERIES (Real-score surrogate + interpretable)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        # ── Full real score history W3-W9 ──
        all_scores = {
            "f1": [6.36e-25, 2.33e-25, 1.19e-26, 2.64e-27, 1.42e-27, 8.76e-28, 2.68e-9],
            "f2": [0.7247,   0.4346,   0.5414,   0.5306,   0.5819,   0.5375,   0.5730],
            "f3": [-0.1268,  -0.1590,  -0.1790,  -0.1701,  -0.1620,  -0.1587,  -0.0585],
            "f4": [-4.378,   -4.378,   -6.056,   -6.254,   -6.379,   -6.494,   -3.986],
            "f5": [445.89,   1320.26,  1623.03,  658.43,   542.69,   520.50,   1189.00],
            "f6": [-0.5522,  -0.6107,  -0.6126,  -0.5762,  -0.5812,  -0.5800,  -0.7636],
            "f7": [1.1865,   1.1880,   1.1862,   1.1852,   1.1848,   1.1843,   1.1841],
            "f8": [8.0703,   8.0724,   8.0712,   8.0712,   8.0598,   8.0593,   8.0590],
        }

        # All rounds W3-W9
        all_rounds = [round3, round4, round5, round6, round7, round8, round9]

        def build_model(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 1), nn.Sigmoid()
            )

        def train_model(model, X_t, y_t, lr=0.01, epochs=5000):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            for _ in range(epochs):
                opt.zero_grad()
                loss_fn(model(X_t), y_t).backward()
                opt.step()
            return model

        def normalise_scores(scores):
            """Normalise real scores to [0,1] for surrogate training."""
            s = np.array(scores, dtype=np.float32)
            s_min, s_max = s.min(), s.max()
            if s_max - s_min < 1e-8:
                return np.full_like(s, 0.5)
            return (s - s_min) / (s_max - s_min)

        result = {}

        for key in round1:
            scores  = all_scores[key]
            points  = [np.array(r[key], dtype=np.float32) for r in all_rounds]
            dim     = len(points[0])
            current = points[-1].copy()  # Round 9 point

            best_idx   = int(np.argmax(scores))
            best_score = scores[best_idx]
            best_point = points[best_idx]
            prev_score = scores[-2]   # W8
            curr_score = scores[-1]   # W9
            trend      = curr_score - prev_score

            # Boundary correction
            for d in range(dim):
                if current[d] <= 0.01: current[d] = 0.05
                elif current[d] >= 0.99: current[d] = 0.95

            print(f"\n{'─'*50}")
            print(f"  {key}: W9={curr_score:.4f} | best=W{best_idx+3}"
                  f"({best_score:.4f}) | trend={trend:+.4f}")

            # ── Train surrogate with REAL normalised scores ──
            norm_labels = normalise_scores(scores)
            X_np = np.array(points, dtype=np.float32)
            y_np = norm_labels.reshape(-1, 1)
            X_t  = torch.tensor(X_np)
            y_t  = torch.tensor(y_np)
            torch.manual_seed(42)
            model = build_model(dim)
            model = train_model(model, X_t, y_t)

            # ── Get surrogate gradient at current point ──
            x_q = torch.tensor(current, dtype=torch.float32, requires_grad=True)
            model(x_q.unsqueeze(0)).backward()
            grad = x_q.grad.numpy()
            grad_norm = grad / (np.linalg.norm(grad) + 1e-8)

            # ── Per-function interpretable strategy ──
            if key == "f1":
                # Score ~0 everywhere. W9 reset to centre showed
                # tiny improvement (2.7e-9 vs 8.7e-28). Try a
                # surrogate-guided step from centre.
                step_size = 0.08
                p10 = np.clip(current + step_size * grad_norm, 0.0, 1.0)
                reason = "surrogate gradient from centre (score still ~0, exploring)"

            elif key == "f2":
                # W9=0.573, improving from W8=0.537.
                # Continue in same direction as W8→W9 move.
                last_move = points[-1] - points[-2]
                move_norm = last_move / (np.linalg.norm(last_move) + 1e-8)
                # Blend with surrogate gradient 50/50
                combined = 0.5 * grad_norm + 0.5 * move_norm
                combined /= (np.linalg.norm(combined) + 1e-8)
                step_size = np.linalg.norm(last_move) * 0.8
                p10 = np.clip(current + step_size * combined, 0.0, 1.0)
                reason = "continue W8→W9 direction + surrogate gradient (improving)"

            elif key == "f3":
                # W9=-0.058, best score ever for f3.
                # Reset to centre worked. Continue surrogate gradient.
                step_size = 0.06
                p10 = np.clip(current + step_size * grad_norm, 0.0, 1.0)
                reason = "surrogate gradient (W9 best ever for f3, keep recovering)"

            elif key == "f4":
                # W9=-3.986, improving from W8=-6.494.
                # Hard reset to centre worked. Surrogate gradient from here.
                step_size = 0.08
                p10 = np.clip(current + step_size * grad_norm, 0.0, 1.0)
                reason = "surrogate gradient from centre (reset worked, continuing recovery)"

            elif key == "f5":
                # W9=1189, moving in right direction toward W5 peak (1623).
                # Continue in same direction — we are recovering well.
                last_move = points[-1] - points[-2]
                move_norm = last_move / (np.linalg.norm(last_move) + 1e-8)
                # Also blend with direction toward all-time best (W5)
                to_best = best_point - current
                to_best_norm = to_best / (np.linalg.norm(to_best) + 1e-8)
                combined = 0.5 * move_norm + 0.5 * to_best_norm
                combined /= (np.linalg.norm(combined) + 1e-8)
                step_size = np.linalg.norm(last_move) * 1.2
                p10 = np.clip(current + step_size * combined, 0.0, 1.0)
                reason = "continue toward W5 best (1623) — recovery on track"

            elif key == "f6":
                # W9=-0.764, reset made it worse.
                # Return toward pre-reset best region (W7=-0.581).
                target = np.array(round7[key], dtype=np.float32)
                direction = target - current
                dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
                step_size = np.linalg.norm(direction) * 0.6
                p10 = np.clip(current + step_size * dir_norm, 0.0, 1.0)
                reason = "return toward W7 region (reset hurt f6, recovering)"

            elif key in ["f7", "f8"]:
                # Both stable and positive. Tiny surrogate gradient step.
                recent = [np.linalg.norm(points[i]-points[i-1])
                          for i in range(4, len(points))]
                step_size = np.mean(recent) * 0.4
                p10 = np.clip(current + step_size * grad_norm, 0.0, 1.0)
                reason = "tiny surrogate gradient (stable positive, careful exploitation)"

            else:
                p10 = current

            result[key] = [round(float(x), 6) for x in p10]
            print(f"  Reason: {reason}")

        lines = []
        for key in result:
            portal_format = "-".join(f"{x:.6f}" for x in result[key])
            print(f"\n{key}:")
            print(f"  Round 9       : {round9[key]}")
            print(f"  Round 10      : {result[key]}")
            print(f"  Portal format : {portal_format}")
            lines.append(f"{key}: {portal_format}")
        _save(lines, "queries_round10.txt")

    except ImportError:
        print("\n  PyTorch not found. Install with: pip install torch")
        print("  Then re-run: python main.py --round 10")



# ═══════════════════════════════════════════════════════════════
# ROUND 11 — Clustering-based strategy (K-means on score history)
#
# Module focus: clustering, distance metrics, centroids
#
# Strategy:
#   For each function we now have 8 real scored points (W3-W10).
#   We apply score-weighted K-means clustering to identify
#   which regions of input space produced the best outputs.
#
#   - Split points into "high" and "low" score clusters
#   - Compute the centroid of the high-score cluster
#   - Query toward that centroid, blended with the best
#     individual point to avoid overshooting
#
#   For functions with clear directional trends (f5, f2, f3, f6)
#   we also incorporate the gradient from the scored surrogate.
#
#   Per-function cluster analysis:
#   f1: all scores ~0, no meaningful cluster — explore randomly
#   f2: two clusters — high (~0.72, W3) and low (~0.43, W4)
#   f3: improving cluster around centre region (W9, W10 best)
#   f4: improving cluster around centre (W9, W10 best)
#   f5: high-score cluster near [0.0-0.2, ~1.0, ~1.0, 0.2-0.4]
#   f6: best cluster near pre-reset W7 region
#   f7: tight cluster around current coords (very stable)
#   f8: tight cluster around current coords (very stable)
# ═══════════════════════════════════════════════════════════════

def run_round11():
    print("\n" + "=" * 60)
    print("ROUND 11 PROPOSED QUERIES (Clustering-based strategy)")
    print("=" * 60)

    # Full real score history W3-W10
    all_scores = {
        "f1": [6.36e-25, 2.33e-25, 1.19e-26, 2.64e-27, 1.42e-27, 8.76e-28, 2.68e-9,  3.22e-14],
        "f2": [0.7247,   0.4346,   0.5414,   0.5306,   0.5819,   0.5375,   0.5730,   0.6305],
        "f3": [-0.1268,  -0.1590,  -0.1790,  -0.1701,  -0.1620,  -0.1587,  -0.0585,  -0.0472],
        "f4": [-4.378,   -4.378,   -6.056,   -6.254,   -6.379,   -6.494,   -3.986,   -4.192],
        "f5": [445.89,   1320.26,  1623.03,  658.43,   542.69,   520.50,   1189.00,  1668.88],
        "f6": [-0.5522,  -0.6107,  -0.6126,  -0.5762,  -0.5812,  -0.5800,  -0.7636,  -0.5182],
        "f7": [1.1865,   1.1880,   1.1862,   1.1852,   1.1848,   1.1843,   1.1841,   1.1841],
        "f8": [8.0703,   8.0724,   8.0712,   8.0712,   8.0598,   8.0593,   8.0590,   8.0590],
    }

    # Rounds W3-W10
    scored_rounds = [round3, round4, round5, round6, round7, round8, round9, round10]

    def score_weighted_centroid(points, scores):
        """
        Compute score-weighted centroid of a set of points.
        Points with higher scores contribute more to the centroid.
        Negative scores get zero weight (they pull away from good regions).
        This is the clustering equivalent of a weighted mean — the
        centroid of the 'high performance cluster'.
        """
        pts = np.array(points, dtype=np.float32)
        s = np.array(scores, dtype=np.float32)
        # Shift scores so minimum = 0 (all weights non-negative)
        s_shifted = s - s.min()
        if s_shifted.sum() < 1e-8:
            return pts.mean(axis=0)  # uniform if all scores equal
        weights = s_shifted / s_shifted.sum()
        return (weights[:, None] * pts).sum(axis=0)

    def high_score_cluster_centroid(points, scores, top_k=3):
        """
        Take the top-k scoring points and compute their centroid.
        This identifies the 'best cluster' in the search history
        and returns its centre — the most promising region found so far.
        """
        pts = np.array(points, dtype=np.float32)
        s   = np.array(scores)
        top_idx = np.argsort(s)[-top_k:]
        return pts[top_idx].mean(axis=0)

    result = {}

    for key in round1:
        scores  = all_scores[key]
        points  = [np.array(r[key], dtype=np.float32) for r in scored_rounds]
        dim     = len(points[0])
        current = np.array(round10[key], dtype=np.float32)

        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_point = points[best_idx]
        curr_score = scores[-1]

        # Boundary correction on current
        for d in range(dim):
            if current[d] <= 0.01: current[d] = 0.05
            elif current[d] >= 0.99: current[d] = 0.95

        # Score-weighted centroid (full history)
        sw_centroid = score_weighted_centroid(points, scores)

        # High-score cluster centroid (top 3 points)
        hs_centroid = high_score_cluster_centroid(points, scores, top_k=3)

        print(f"\n{'─'*50}")
        print(f"  {key}: W10={curr_score:.4f} | best=W{best_idx+3}({best_score:.4f})")
        print(f"  Score-weighted centroid : {[round(float(x),4) for x in sw_centroid]}")
        print(f"  High-score centroid     : {[round(float(x),4) for x in hs_centroid]}")

        if key == "f1":
            # All scores ~0 — no meaningful cluster.
            # Try a different unexplored region: move toward
            # the corner opposite to current position.
            opposite = np.clip(1.0 - current, 0.05, 0.95)
            p11 = opposite
            cluster = "no cluster (all ~0) — explore opposite corner"

        elif key == "f2":
            # Best score at W3 (0.7247), W10=0.631 improving.
            # High-score cluster centroid points toward W3 region.
            # Step 70% toward high-score centroid from current.
            direction = hs_centroid - current
            p11 = np.clip(current + 0.7 * direction, 0.0, 1.0)
            cluster = f"high-score cluster centroid (top-3 avg, best=W{best_idx+3})"

        elif key == "f3":
            # W10=-0.047 best ever. Recovery continuing.
            # Score-weighted centroid incorporates recent improvements.
            # Step 60% toward score-weighted centroid.
            direction = sw_centroid - current
            p11 = np.clip(current + 0.6 * direction, 0.0, 1.0)
            cluster = "score-weighted centroid (recent recovery cluster)"

        elif key == "f4":
            # W10=-4.19, slight dip from W9=-3.99.
            # Best recent cluster is around centre.
            # High-score centroid points toward least-bad region.
            direction = hs_centroid - current
            p11 = np.clip(current + 0.5 * direction, 0.0, 1.0)
            cluster = "high-score centroid (least-negative cluster)"

        elif key == "f5":
            # W10=1668 — new all-time best! Dims 2&3 at 1.0.
            # The high-score cluster is clearly near [low, ~1, ~1, mid].
            # Continue pushing: keep dims 2&3 near 1.0,
            # use centroid to guide dims 1&4.
            direction = hs_centroid - current
            dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
            step_size = np.linalg.norm(direction) * 0.4
            p11_raw = current + step_size * dir_norm
            # Force dims 2&3 to stay high — the cluster confirms this
            p11_raw[1] = min(1.0, p11_raw[1] + 0.02)
            p11_raw[2] = min(1.0, p11_raw[2] + 0.02)
            p11 = np.clip(p11_raw, 0.0, 1.0)
            cluster = "high-score cluster [low,~1,~1,mid] — exploit peak"

        elif key == "f6":
            # W10=-0.518 best ever. Score-weighted centroid
            # should now reflect the recovering region.
            direction = sw_centroid - current
            p11 = np.clip(current + 0.6 * direction, 0.0, 1.0)
            cluster = "score-weighted centroid (best-ever W10, keep recovering)"

        elif key in ["f7", "f8"]:
            # Extremely tight cluster — all points near same coords.
            # Centroid is essentially the current point.
            # Tiny step toward score-weighted centroid (best cluster).
            direction = sw_centroid - current
            step_size = np.linalg.norm(direction) * 0.3
            dir_norm = direction / (np.linalg.norm(direction) + 1e-8)
            p11 = np.clip(current + step_size * dir_norm, 0.0, 1.0)
            cluster = "tight stable cluster — micro-step toward centroid"

        else:
            p11 = current

        result[key] = [round(float(x), 6) for x in p11]
        print(f"  Cluster strategy: {cluster}")

    lines = []
    for key in result:
        portal_format = "-".join(f"{x:.6f}" for x in result[key])
        print(f"\n{key}:")
        print(f"  Round 10      : {round10[key]}")
        print(f"  Round 11      : {result[key]}")
        print(f"  Portal format : {portal_format}")
        lines.append(f"{key}: {portal_format}")
    _save(lines, "queries_round11.txt")



# ═══════════════════════════════════════════════════════════════
# ROUND 12 — PCA-guided search
#
# Module focus: Principal Component Analysis, dimensionality
# reduction, variance explanation
#
# Strategy:
#   For each function we have 8 real scored points (W3-W10).
#   We apply PCA to the input coordinates of those points to
#   find the principal directions of variation in the search
#   space. The first principal component (PC1) captures the
#   direction of maximum variance across all queries.
#
#   We then correlate each PC with score improvements to find
#   the "score-aligned principal direction" — the direction
#   in input space that explains the most variance AND points
#   toward higher scores.
#
#   This is more principled than gradient-based search because:
#   - PCA uses ALL historical data, not just the current point
#   - It identifies structure in the WHOLE search trajectory
#   - It separates meaningful variation (score-correlated PCs)
#     from noise (low-variance PCs)
#
#   Per-function strategy:
#   f1: PCA on 2D space — find unexplored PC direction
#   f2: PC aligned with score increase direction
#   f3: Continue recovery using score-aligned PC
#   f4: PC from centre region (post-reset cluster)
#   f5: PC1 in 4D space — dims 2&3 dominate, stay high
#   f6: Score-aligned PC from recovery cluster
#   f7: Micro-step along PC1 (tight cluster)
#   f8: Micro-step along PC1 (tight cluster)
# ═══════════════════════════════════════════════════════════════

def run_round12():
    print("\n" + "=" * 60)
    print("ROUND 12 PROPOSED QUERIES (PCA-guided search)")
    print("=" * 60)

    # Real score history W3-W10 (8 points)
    all_scores = {
        "f1": [6.36e-25, 2.33e-25, 1.19e-26, 2.64e-27, 1.42e-27, 8.76e-28, 2.68e-9,  3.22e-14],
        "f2": [0.7247,   0.4346,   0.5414,   0.5306,   0.5819,   0.5375,   0.5730,   0.6305],
        "f3": [-0.1268,  -0.1590,  -0.1790,  -0.1701,  -0.1620,  -0.1587,  -0.0585,  -0.0472],
        "f4": [-4.378,   -4.378,   -6.056,   -6.254,   -6.379,   -6.494,   -3.986,   -4.192],
        "f5": [445.89,   1320.26,  1623.03,  658.43,   542.69,   520.50,   1189.00,  1668.88],
        "f6": [-0.5522,  -0.6107,  -0.6126,  -0.5762,  -0.5812,  -0.5800,  -0.7636,  -0.5182],
        "f7": [1.1865,   1.1880,   1.1862,   1.1852,   1.1848,   1.1843,   1.1841,   1.1841],
        "f8": [8.0703,   8.0724,   8.0712,   8.0712,   8.0598,   8.0593,   8.0590,   8.0590],
    }

    # Scored rounds W3-W10
    scored_rounds = [round3, round4, round5, round6, round7, round8, round9, round10]

    def pca_directions(points):
        """
        Compute principal components of the query history.
        Centers the data, computes covariance matrix, returns
        eigenvectors sorted by descending eigenvalue.

        PC1 = direction of maximum variance in the search history
        PC2 = direction of second most variance, orthogonal to PC1
        etc.
        """
        X = np.array(points, dtype=np.float64)
        mean = X.mean(axis=0)
        X_centered = X - mean
        cov = np.cov(X_centered.T)
        if cov.ndim == 0:
            return np.array([[1.0]]), mean
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx], mean, eigenvalues[idx]

    def score_aligned_pc(points, scores, pcs):
        """
        Find which principal component direction is most aligned
        with score improvement. For each PC, check if moving in
        that direction correlates positively with score.

        Returns the PC direction (or its negative) that points
        toward higher scores — the 'score-aligned principal direction'.
        """
        pts   = np.array(points, dtype=np.float64)
        s     = np.array(scores, dtype=np.float64)
        mean  = pts.mean(axis=0)
        X_c   = pts - mean

        best_corr = -np.inf
        best_dir  = pcs[:, 0]

        for i in range(pcs.shape[1]):
            pc = pcs[:, i]
            # Project each point onto this PC
            projections = X_c @ pc
            # Correlation between projection and score
            if np.std(projections) < 1e-10:
                continue
            corr = np.corrcoef(projections, s)[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                # Flip direction if negatively correlated
                best_dir = pc if corr > 0 else -pc

        return best_dir, best_corr

    result = {}

    for key in round1:
        scores  = all_scores[key]
        points  = [np.array(r[key], dtype=np.float32) for r in scored_rounds]
        dim     = len(points[0])
        current = np.array(round11[key], dtype=np.float32)

        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]
        curr_score = scores[-1]  # W10

        # Boundary correction
        for d in range(dim):
            if current[d] <= 0.01: current[d] = 0.05
            elif current[d] >= 0.99: current[d] = 0.95

        # Compute PCA on scored history
        pcs, mean, eigenvalues = pca_directions(points)
        variance_explained = eigenvalues / (eigenvalues.sum() + 1e-10)

        # Find score-aligned principal direction
        score_dir, corr = score_aligned_pc(points, scores, pcs)
        score_dir = score_dir.astype(np.float32)

        print(f"\n{'─'*50}")
        print(f"  {key}: W10={curr_score:.4f} | best=W{best_idx+3}({best_score:.4f})")
        print(f"  PC1 variance explained : {variance_explained[0]:.1%}")
        print(f"  Score-PC correlation   : {corr:.3f}")

        if key == "f1":
            # All scores ~0. PCA reveals the main direction of
            # variance in our search trajectory. Explore orthogonal
            # to where we've already been (PC2 = unexplored direction).
            if pcs.shape[1] > 1:
                unexplored = pcs[:, 1].astype(np.float32)
            else:
                unexplored = score_dir
            step_size = 0.12
            p12 = np.clip(current + step_size * unexplored, 0.0, 1.0)
            reason = f"PC2 direction (unexplored axis, PC1 var={variance_explained[0]:.1%})"

        elif key == "f2":
            # Score-aligned PC points toward higher-score region.
            # Step along it from current point.
            recent_steps = [np.linalg.norm(np.array(scored_rounds[i][key]) -
                            np.array(scored_rounds[i-1][key]))
                            for i in range(1, len(scored_rounds))]
            step_size = np.mean(recent_steps) * 0.6
            p12 = np.clip(current + step_size * score_dir, 0.0, 1.0)
            reason = f"score-aligned PC (corr={corr:.3f})"

        elif key == "f3":
            # Recovery continuing. Score-aligned PC captures the
            # direction of improvement toward centre region.
            step_size = 0.05
            p12 = np.clip(current + step_size * score_dir, 0.0, 1.0)
            reason = f"score-aligned PC (recovery direction, corr={corr:.3f})"

        elif key == "f4":
            # Best cluster is around centre. Score-aligned PC
            # from post-reset data points toward least-bad region.
            step_size = 0.07
            p12 = np.clip(current + step_size * score_dir, 0.0, 1.0)
            reason = f"score-aligned PC from centre cluster (corr={corr:.3f})"

        elif key == "f5":
            # f5 PC1 will be dominated by dims 2&3 (most variance,
            # highest scores when near 1.0). Step strongly along
            # score-aligned PC — this is f5's peak direction.
            recent_steps = [np.linalg.norm(np.array(scored_rounds[i][key]) -
                            np.array(scored_rounds[i-1][key]))
                            for i in range(1, len(scored_rounds))]
            step_size = np.mean(recent_steps) * 0.5
            p12_raw = current + step_size * score_dir
            # Keep dims 2&3 high — cluster and PCA both confirm this
            p12_raw[1] = max(p12_raw[1], 0.95)
            p12_raw[2] = max(p12_raw[2], 0.95)
            p12 = np.clip(p12_raw, 0.0, 1.0)
            reason = f"score-aligned PC1 (var={variance_explained[0]:.1%}, corr={corr:.3f})"

        elif key == "f6":
            # Best-ever at W10. Score-aligned PC should reflect
            # the improvement direction from the recovery.
            step_size = 0.05
            p12 = np.clip(current + step_size * score_dir, 0.0, 1.0)
            reason = f"score-aligned PC (best-ever W10, corr={corr:.3f})"

        elif key in ["f7", "f8"]:
            # Extremely tight cluster. PC1 variance explains nearly
            # all variation. Micro-step along score-aligned PC.
            step_size = 0.002
            p12 = np.clip(current + step_size * score_dir, 0.0, 1.0)
            reason = f"micro-step along score-aligned PC (tight cluster, corr={corr:.3f})"

        else:
            p12 = current

        result[key] = [round(float(x), 6) for x in p12]
        print(f"  Reason: {reason}")

    lines = []
    for key in result:
        portal_format = "-".join(f"{x:.6f}" for x in result[key])
        print(f"\n{key}:")
        print(f"  Round 11      : {round11[key]}")
        print(f"  Round 12      : {result[key]}")
        print(f"  Portal format : {portal_format}")
        lines.append(f"{key}: {portal_format}")
    _save(lines, "queries_round12.txt")



# ═══════════════════════════════════════════════════════════════
# ROUND 13 — FINAL ROUND
# Reinforcement Learning lens: MAB, Q-learning, MDP
#
# This is the last query. No recovery is possible.
# Strategy: pure exploitation of confirmed best regions.
#
# RL parallels:
#   - MAB (Multi-Armed Bandit): each function is an "arm".
#     With the final pull, we choose the action with the
#     highest estimated Q-value based on all observed rewards.
#   - Q-learning: our score history is the Q-table.
#     The greedy policy selects argmax Q(s, a) — the best
#     known action from the best known state.
#   - MDP feedback-driven adaptation: the full trajectory
#     is our experience replay. Round 13 uses the complete
#     history to select the final policy action.
#
# Full real score history W3-W12 (W11 estimated):
#   f1: ~0 everywhere — submit near best centroid
#   f2: best W3 (0.7247), W12=0.605, dipping — return to W3 region
#   f3: best W10 (-0.047), W12=-0.083 dipping — return to W10 point
#   f4: best W9 (-3.986), W12=-4.122 slight dip — return to W9 point
#   f5: best W10 (1668), W12=1412 — overshot, pull back dims 2&3
#   f6: best W10 (-0.518), W12=-0.556 — return to W10 point
#   f7: best W4 (1.188), W12=1.182 tiny dip — return to W4 region
#   f8: best W4 (8.072), W12=8.066 — submit W4 point (best ever)
# ═══════════════════════════════════════════════════════════════

def run_round13():
    print("\n" + "=" * 60)
    print("ROUND 13 — FINAL ROUND (RL-informed pure exploitation)")
    print("=" * 60)

    # Full score history W3-W12 (W11 interpolated)
    all_scores = {
        "f1": [6.36e-25, 2.33e-25, 1.19e-26, 2.64e-27, 1.42e-27, 8.76e-28, 2.68e-9,  3.22e-14, 2.68e-9,  3.80e-35],
        "f2": [0.7247,   0.4346,   0.5414,   0.5306,   0.5819,   0.5375,   0.5730,   0.6305,   0.6180,   0.6050],
        "f3": [-0.1268,  -0.1590,  -0.1790,  -0.1701,  -0.1620,  -0.1587,  -0.0585,  -0.0472,  -0.0651,  -0.0829],
        "f4": [-4.378,   -4.378,   -6.056,   -6.254,   -6.379,   -6.494,   -3.986,   -4.192,   -4.157,   -4.122],
        "f5": [445.89,   1320.26,  1623.03,  658.43,   542.69,   520.50,   1189.00,  1668.88,  1540.66,  1412.44],
        "f6": [-0.5522,  -0.6107,  -0.6126,  -0.5762,  -0.5812,  -0.5800,  -0.7636,  -0.5182,  -0.5371,  -0.5560],
        "f7": [1.1865,   1.1880,   1.1862,   1.1852,   1.1848,   1.1843,   1.1841,   1.1841,   1.1831,   1.1822],
        "f8": [8.0703,   8.0724,   8.0712,   8.0712,   8.0598,   8.0593,   8.0590,   8.0590,   8.0623,   8.0656],
    }

    # All scored rounds W3-W12
    scored_rounds = [round3, round4, round5, round6, round7,
                     round8, round9, round10, round11, round12]

    result = {}

    for key in round1:
        scores = all_scores[key]
        points = [np.array(r[key], dtype=np.float32) for r in scored_rounds]
        dim    = len(points[0])

        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_point = points[best_idx].copy()
        curr_score = scores[-1]

        print(f"\n{'─'*50}")
        print(f"  {key}: best=W{best_idx+3}({best_score:.4f}) | W12={curr_score:.4f}")

        if key == "f1":
            # All scores ~0, best was W9 (2.68e-9).
            # Submit score-weighted centroid of top-3 points.
            top_idx = np.argsort(scores)[-3:]
            p13 = np.array([points[i] for i in top_idx]).mean(axis=0)
            reason = "centroid of top-3 points (best available for flat function)"

        elif key == "f2":
            # Best was W3 (0.7247). W12 dipping.
            # Submit W3 point directly — highest ever observed.
            p13 = best_point
            reason = f"W3 best point (0.7247, highest ever observed)"

        elif key == "f3":
            # Best was W10 (-0.047). W12 dipping back.
            # Submit W10 point directly.
            p13 = best_point
            reason = f"W10 best point (-0.047, best ever for f3)"

        elif key == "f4":
            # Best was W9 (-3.986). W12=-4.122, dipping.
            # Submit W9 point directly.
            p13 = best_point
            reason = f"W9 best point (-3.986, best ever for f4)"

        elif key == "f5":
            # Best W10 (1668). W12=1412 — overshot as dims 2&3
            # pushed too close to 1.0. Pull them back slightly.
            # Submit W10 point but with dims 2&3 at 0.96 instead of ~1.0
            p13 = best_point.copy()
            p13[1] = min(p13[1], 0.960)  # pull dim2 back from boundary
            p13[2] = min(p13[2], 0.960)  # pull dim3 back from boundary
            p13 = np.clip(p13, 0.0, 1.0)
            reason = f"W10 best point with dims 2&3 pulled to 0.96 (overshot correction)"

        elif key == "f6":
            # Best W10 (-0.518). W12=-0.556 dipping.
            # Submit W10 point directly.
            p13 = best_point
            reason = f"W10 best point (-0.518, best ever for f6)"

        elif key == "f7":
            # Best W4 (1.188). Very stable.
            # Submit W4 point — best ever observed.
            p13 = best_point
            reason = f"W4 best point (1.1880, best ever for f7)"

        elif key == "f8":
            # Best W4 (8.0724). W12=8.0656, slight improvement trend.
            # Submit W4 point — best ever observed.
            p13 = best_point
            reason = f"W4 best point (8.0724, best ever for f8)"

        else:
            p13 = points[-1]

        result[key] = [round(float(x), 6) for x in p13]
        print(f"  Strategy: {reason}")

    lines = []
    for key in result:
        portal_format = "-".join(f"{x:.6f}" for x in result[key])
        print(f"\n{key}:")
        print(f"  Round 12      : {round12[key]}")
        print(f"  Round 13      : {result[key]}")
        print(f"  Portal format : {portal_format}")
        lines.append(f"{key}: {portal_format}")
    _save(lines, "queries_round13.txt")


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
        "--round", type=int, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        help="Which round to run (2-13). Omit to run all."
    )
    args = parser.parse_args()

    if args.round == 2:   run_round2()
    elif args.round == 3: run_round3()
    elif args.round == 4: run_round4()
    elif args.round == 5: run_round5()
    elif args.round == 6: run_round6()
    elif args.round == 7: run_round7()
    elif args.round == 8: run_round8()
    elif args.round == 9: run_round9()
    elif args.round == 10: run_round10()
    elif args.round == 11: run_round11()
    elif args.round == 12: run_round12()
    elif args.round == 13: run_round13()
    else:
        run_round2(); run_round3(); run_round4()
        run_round5(); run_round6(); run_round7()
        run_round8(); run_round9(); run_round10()
        run_round11(); run_round12(); run_round13()