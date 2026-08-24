"""
A small neural network, built by hand with numpy (forward + backward pass,
Adam optimizer) - consistent with how everything else in this project
(Q-learning, MCTS, the solver, TD(0)) was built from first principles rather
than using a library.

This is the piece that actually closes the biggest gap with AlphaZero: our
Q-table can only ever answer questions about states it has EXACTLY visited
before; a network generalizes to positions it has never seen, based on
similarity to positions it has.

Input encoding (19 numbers) - always from the CURRENT PLAYER's perspective,
the same "canonicalization" trick AlphaZero uses, so the network never has
to separately learn "I am X" vs "I am O" as different concepts:
  - 9 numbers: 1.0 where the current player has a piece, else 0.0
  - 9 numbers: 1.0 where the opponent has a piece, else 0.0
  - 1 number: 0.0 if placement phase, 1.0 if sliding phase

Two output heads, sharing the same hidden layers:
  - Value head: one number in [-1, 1] (tanh) - "how good is this position
    for whoever is about to move" - directly comparable to the solver's
    own negamax_value convention used throughout this project.
  - Policy head: 90 logits - 9 placement actions (place at cell i) plus
    81 sliding actions (from*9 + to, covering every from/to combination;
    illegal ones are masked out whenever the policy is actually used).
"""

import pickle

import numpy as np

INPUT_SIZE = 19
POLICY_SIZE = 90  # 9 placement actions + 9*9 (from, to) sliding actions


def encode_state(board, phase, current_player):
    """Board -> the 19-number input vector, from current_player's perspective."""
    x = np.zeros(INPUT_SIZE, dtype=np.float64)
    for i, cell in enumerate(board):
        if cell == current_player:
            x[i] = 1.0
        elif cell != 0:
            x[9 + i] = 1.0
    x[18] = 1.0 if phase == 'sliding' else 0.0
    return x


def encode_action(move):
    """A placement move (int) or sliding move (from, to) -> an index into
    the 90-slot policy vector."""
    if isinstance(move, tuple):
        from_cell, to_cell = move
        return 9 + from_cell * 9 + to_cell
    return move


def decode_action_index(index):
    """Inverse of encode_action - an index -> a placement int or (from, to) tuple."""
    if index < 9:
        return index
    remainder = index - 9
    return (remainder // 9, remainder % 9)


class TicTacToeNet:
    """Two hidden layers (ReLU), shared trunk, two output heads. Trained
    with plain mini-batch gradient descent + Adam, both hand-written."""

    def __init__(self, hidden_size=64, seed=0):
        rng = np.random.default_rng(seed)

        def init_layer(fan_in, fan_out):
            # He initialization - a standard, sensible default for ReLU nets
            return rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

        self.W1 = init_layer(INPUT_SIZE, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = init_layer(hidden_size, hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W_value = init_layer(hidden_size, 1)
        self.b_value = np.zeros(1)
        self.W_policy = init_layer(hidden_size, POLICY_SIZE)
        self.b_policy = np.zeros(POLICY_SIZE)

        self.params = ['W1', 'b1', 'W2', 'b2', 'W_value', 'b_value', 'W_policy', 'b_policy']
        # Adam optimizer state
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, x_batch):
        """x_batch: (N, 19). Returns (value (N,1), policy_logits (N,90)) plus
        the intermediate activations needed for backprop."""
        z1 = x_batch @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU

        value = np.tanh(a2 @ self.W_value + self.b_value)
        policy_logits = a2 @ self.W_policy + self.b_policy

        cache = (x_batch, z1, a1, z2, a2, value)
        return value, policy_logits, cache

    def predict(self, board, phase, current_player):
        """Single-position convenience wrapper: returns (value, policy_probs_over_90)."""
        x = encode_state(board, phase, current_player).reshape(1, -1)
        value, policy_logits, _ = self.forward(x)
        policy_probs = _softmax(policy_logits[0])
        return float(value[0, 0]), policy_probs

    def train_step(self, x_batch, target_value, target_policy, legal_mask, learning_rate=0.001):
        """One gradient step on a batch.
        target_value: (N,1) in [-1,1] - the real outcome (or solver value).
        target_policy: (N,90) - a probability distribution per example (e.g.
            MCTS visit counts normalized to sum to 1).
        legal_mask: (N,90) of 0/1 - illegal actions never contribute to the
            policy loss or its gradient.
        Returns (value_loss, policy_loss) for monitoring.
        """
        n = x_batch.shape[0]
        value, policy_logits, (x, z1, a1, z2, a2, _) = self.forward(x_batch)

        # ---- Value loss: mean squared error ----
        value_error = value - target_value  # (N,1)
        value_loss = float(np.mean(value_error ** 2))

        # ---- Policy loss: cross-entropy against target distribution,
        # restricted to legal actions (illegal logits are masked to -inf
        # before softmax so they get zero probability and zero gradient) ----
        masked_logits = np.where(legal_mask > 0, policy_logits, -1e9)
        policy_probs = _softmax(masked_logits)
        safe_probs = np.clip(policy_probs, 1e-9, 1.0)
        policy_loss = float(-np.mean(np.sum(target_policy * np.log(safe_probs), axis=1)))

        # ---- Backward pass (hand-derived gradients) ----
        d_value_head = value_error * (1 - value ** 2) * (2.0 / n)  # d(loss)/d(pre-tanh), chain through tanh
        d_policy_logits = (policy_probs - target_policy) * legal_mask / n

        d_a2 = d_value_head @ self.W_value.T + d_policy_logits @ self.W_policy.T
        d_W_value = a2.T @ d_value_head
        d_b_value = d_value_head.sum(axis=0)
        d_W_policy = a2.T @ d_policy_logits
        d_b_policy = d_policy_logits.sum(axis=0)

        d_z2 = d_a2 * (z2 > 0)
        d_W2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (z1 > 0)
        d_W1 = x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        grads = {
            'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2,
            'W_value': d_W_value, 'b_value': d_b_value,
            'W_policy': d_W_policy, 'b_policy': d_b_policy,
        }
        self._adam_step(grads, learning_rate)

        return value_loss, policy_loss

    def train_value_step(self, x_batch, target_value, learning_rate=0.001):
        """Train ONLY the value head (used to validate, in isolation, that the
        network genuinely generalizes to unseen states - no policy target
        exists yet at this stage of the project, so this avoids polluting the
        shared trunk with a meaningless policy gradient)."""
        n = x_batch.shape[0]
        value, _, (x, z1, a1, z2, a2, _) = self.forward(x_batch)

        value_error = value - target_value
        value_loss = float(np.mean(value_error ** 2))

        d_value_head = value_error * (1 - value ** 2) * (2.0 / n)
        d_W_value = a2.T @ d_value_head
        d_b_value = d_value_head.sum(axis=0)

        d_a2 = d_value_head @ self.W_value.T
        d_z2 = d_a2 * (z2 > 0)
        d_W2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (z1 > 0)
        d_W1 = x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        grads = {
            'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2,
            'W_value': d_W_value, 'b_value': d_b_value,
            'W_policy': np.zeros_like(self.W_policy), 'b_policy': np.zeros_like(self.b_policy),
        }
        self._adam_step(grads, learning_rate)
        return value_loss

    def _adam_step(self, grads, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def save(self, filename):
        data = {p: getattr(self, p) for p in self.params}
        data['t'] = self.t
        data['m'] = self.m
        data['v'] = self.v
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        for p in self.params:
            setattr(self, p, data[p])
        self.t = data['t']
        self.m = data['m']
        self.v = data['v']


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
