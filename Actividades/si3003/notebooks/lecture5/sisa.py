"""
Reconocimiento de dígitos escritos a mano usando Reinforcement Learning (REINFORCE).

Idea general:
-------------
En vez de entrenar con aprendizaje supervisado clásico (cross-entropy directo sobre
las etiquetas), planteamos la clasificación como un problema de RL:

    - Estado (s):   la imagen del dígito (vector de píxeles).
    - Acción (a):   el dígito que la "política" decide predecir (0-9).
    - Política π(a|s): una red neuronal que da una distribución de probabilidad
                        sobre los 10 posibles dígitos.
    - Recompensa (r): +1 si la acción elegida coincide con la etiqueta real,
                       -1 si no coincide.
    - Objetivo:     maximizar la recompensa esperada -> aprender a "actuar"
                     (clasificar) cada vez mejor.

Usamos el algoritmo REINFORCE (policy gradient de Monte Carlo):

    grad_theta J = E[ (r - b) * grad_theta log π(a|s) ]

donde b es una línea base (baseline, la media móvil de la recompensa) que reduce
la varianza del gradiente sin introducir sesgo.

Dataset:
--------
Usamos sklearn.datasets.load_digits (imágenes 8x8 = 64 píxeles, 10 clases),
que no requiere descarga de internet y es perfecto para un ejemplo autocontenido.
Todo está implementado con numpy puro (sin frameworks de deep learning) para que
se vea explícitamente cómo funciona el gradiente de política.

Ejecución:
----------
    python3 rl_digit_recognition.py
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# 1. Utilidades
# ----------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable, fila a fila (batch, n_clases)."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def one_hot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], n_classes))
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


# ----------------------------------------------------------------------
# 2. Política (red neuronal de 2 capas) entrenada con REINFORCE
# ----------------------------------------------------------------------

class PolicyNetwork:
    def __init__(self, n_inputs: int, n_hidden: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Inicialización He para la capa oculta (ReLU)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_inputs), size=(n_inputs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_actions))
        self.b2 = np.zeros(n_actions)

    def forward(self, X: np.ndarray):
        """Devuelve activaciones intermedias necesarias para el backward."""
        z1 = X @ self.W1 + self.b1
        h = np.maximum(0, z1)          # ReLU
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)
        return h, probs

    def act(self, X: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        """Muestrea una acción (dígito) por cada estado (imagen) del batch."""
        h, probs = self.forward(X)
        if greedy:
            actions = probs.argmax(axis=1)
        else:
            # Muestreo por fila según la distribución de probabilidad de la política
            cum = probs.cumsum(axis=1)
            u = rng.random((X.shape[0], 1))
            actions = (u < cum).argmax(axis=1)
        return actions, probs, h

    def update(self, X, h, probs, actions, advantages, lr: float):
        """Un paso de ascenso de gradiente de política (REINFORCE)."""
        n_actions = probs.shape[1]
        batch_size = X.shape[0]

        # d log pi(a|s) / d logits = onehot(a) - probs
        # Multiplicamos por la ventaja (recompensa - baseline) para el ascenso.
        dlogits = advantages[:, None] * (one_hot(actions, n_actions) - probs)
        dlogits /= batch_size

        # Backprop manual (ascenso: sumamos el gradiente en vez de restarlo)
        grad_W2 = h.T @ dlogits
        grad_b2 = dlogits.sum(axis=0)

        dh = dlogits @ self.W2.T
        dh[h <= 0] = 0.0  # derivada de ReLU

        grad_W1 = X.T @ dh
        grad_b1 = dh.sum(axis=0)

        self.W1 += lr * grad_W1
        self.b1 += lr * grad_b1
        self.W2 += lr * grad_W2
        self.b2 += lr * grad_b2


# ----------------------------------------------------------------------
# 3. Entrenamiento
# ----------------------------------------------------------------------

def train(
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 0.05,
    baseline_momentum: float = 0.9,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # --- Datos ---
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target  # normalizamos píxeles a [0,1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    policy = PolicyNetwork(n_inputs=64, n_hidden=128, n_actions=10, seed=seed)
    baseline = 0.0  # línea base = recompensa media móvil, reduce varianza

    n_samples = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_samples)
        X_train, y_train = X_train[perm], y_train[perm]

        epoch_rewards = []

        for start in range(0, n_samples, batch_size):
            xb = X_train[start:start + batch_size]
            yb = y_train[start:start + batch_size]

            actions, probs, h = policy.act(xb, rng, greedy=False)
            rewards = np.where(actions == yb, 1.0, -1.0)

            # Actualizamos la línea base (media móvil de la recompensa)
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * rewards.mean()
            advantages = rewards - baseline

            policy.update(xb, h, probs, actions, advantages, lr=lr)
            epoch_rewards.append(rewards.mean())

        # --- Evaluación (política greedy = argmax, sin exploración) ---
        if epoch % 5 == 0 or epoch == 1:
            test_actions, _, _ = policy.act(X_test, rng, greedy=True)
            acc = (test_actions == y_test).mean()
            print(
                f"Época {epoch:3d} | recompensa media (train) = {np.mean(epoch_rewards):+.3f} "
                f"| baseline = {baseline:+.3f} | precisión test = {acc:.3%}"
            )

    # Precisión final
    test_actions, _, _ = policy.act(X_test, rng, greedy=True)
    final_acc = (test_actions == y_test).mean()
    print(f"\nPrecisión final en test: {final_acc:.3%}")
    return policy, final_acc


if __name__ == "__main__":
    train()