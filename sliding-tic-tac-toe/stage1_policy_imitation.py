"""
The untried experiment: broad supervised policy imitation, mirroring exactly
how Stage 1's value-only pretraining worked (research/alphazero_deep_dive/
05_results_summary.md) - direct supervision on a large, random sample of
solved states, with NO self-play in the loop at all - but applied to the
POLICY head instead of just the value head.

Why this is different from the two previous (failed) policy-imitation
attempts (teacher, imitate in alphazero_selfplay.py): those only ever showed
the policy a correct answer at whatever handful of positions self-play
happened to visit - sparse, incidental, gated behind the self-play process
itself ever reaching that state. This script instead samples broadly and
directly from ALL ~4,974 solved states up front, exactly like Stage 1 did
for the value head, which is the one thing in this whole project that
DID generalize to unseen states (0.206 test MAE). This is the natural
extension of that same recipe to the policy head, which nothing has tried yet.

Ties handled correctly: at many positions more than one move is EQUALLY
optimal. Cloning only the solver's single (arbitrarily tie-broken) choice
would incorrectly penalize the network for confidently picking a different,
equally valid optimal move - so the training target here is a uniform
distribution over ALL moves tied for the best value at that position, not
just the first one perfect_solver.best_move() happens to return.
"""

import random

import numpy as np

from game_engine import SlidingTicTacToe, X, O
from neural_net import TicTacToeNet, encode_state, encode_action, POLICY_SIZE
from neural_mcts import neural_mcts_search
from perfect_solver import _enumerate_states, _value_iteration, negamax_value, best_move as perfect_solver_move


def reconstruct_game(board, phase, player):
    """Rebuilds a game object from a (board, phase, player) state key.
    Critically must also set pieces_placed to match the board's actual
    piece counts - leaving it at the default {X: 0, O: 0} would make any
    subsequent make_move() call transition phases incorrectly (e.g. never
    switching to 'sliding' when it should), producing boards that are
    inconsistent with the real state space and absent from the solver's
    enumerated states - the same class of bug hit earlier in this project
    when manually constructing test states without this field."""
    game = SlidingTicTacToe()
    game.board = list(board)
    game.phase = phase
    game.current_player = player
    game.pieces_placed = {
        X: sum(1 for c in board if c == X),
        O: sum(1 for c in board if c == O),
    }
    return game


def all_optimal_moves(game):
    """Every move tied for the maximum value at this position - the fair
    behavioral-cloning target, including ties."""
    valid_moves = game.get_valid_moves()
    best_value = -2.0
    move_values = []
    for move in valid_moves:
        trial = game.copy()
        trial.make_move(move)
        if trial.game_over:
            v = 0.0 if trial.winner == -1 else (1.0 if trial.winner == game.current_player else -1.0)
        else:
            v = -negamax_value(trial)
        move_values.append((move, v))
        best_value = max(best_value, v)
    return [m for m, v in move_values if v >= best_value - 1e-9]


def build_examples(keys):
    examples = []
    for board, phase, player in keys:
        game = reconstruct_game(board, phase, player)
        valid_moves = game.get_valid_moves()
        optimal_moves = all_optimal_moves(game)

        policy_target = np.zeros(POLICY_SIZE)
        for m in optimal_moves:
            policy_target[encode_action(m)] = 1.0 / len(optimal_moves)

        legal_mask = np.zeros(POLICY_SIZE)
        for m in valid_moves:
            legal_mask[encode_action(m)] = 1.0

        x = encode_state(game.board, game.phase, game.current_player)
        value_target = negamax_value(game)
        examples.append((x, policy_target, value_target, legal_mask, optimal_moves))
    return examples


def train_epoch(net, examples, batch_size=64, learning_rate=0.001):
    random.shuffle(examples)
    total_value_loss = 0.0
    total_policy_loss = 0.0
    n_batches = 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        x_batch = np.array([e[0] for e in batch])
        policy_batch = np.array([e[1] for e in batch])
        value_batch = np.array([[e[2]] for e in batch])
        legal_mask_batch = np.array([e[3] for e in batch])
        value_loss, policy_loss = net.train_step(x_batch, value_batch, policy_batch, legal_mask_batch, learning_rate)
        total_value_loss += value_loss
        total_policy_loss += policy_loss
        n_batches += 1
    return total_value_loss / n_batches, total_policy_loss / n_batches


def evaluate(net, examples):
    """Value MAE, and policy top-1 accuracy: does the network's own
    argmax-over-legal-moves choice fall within the set of truly optimal
    moves for that position (accounting for ties)?"""
    value_errors = []
    correct = 0
    for x, policy_target, value_target, legal_mask, optimal_moves in examples:
        v, logits, _ = net.forward(x.reshape(1, -1))
        pred_value = float(v[0, 0])
        masked_logits = np.where(legal_mask > 0, logits[0], -1e9)
        policy_probs = _softmax_np(masked_logits)
        value_errors.append(abs(pred_value - value_target))
        predicted_action = int(np.argmax(policy_probs))
        if predicted_action in [encode_action(m) for m in optimal_moves]:
            correct += 1
    return sum(value_errors) / len(value_errors), correct / len(examples)


def _softmax_np(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def decisive_test(net, iterations=100, games=20):
    def run_side(network_plays):
        wins = losses = draws = 0
        solver_side = O if network_plays == X else X
        for _ in range(games):
            game = SlidingTicTacToe()
            while not game.game_over:
                if game.current_player == solver_side:
                    move = perfect_solver_move(game)
                else:
                    move, _ = neural_mcts_search(game, net, iterations=iterations, add_noise=False)
                game.make_move(move)
            if game.winner == network_plays:
                wins += 1
            elif game.winner == -1:
                draws += 1
            else:
                losses += 1
        return wins, losses, draws

    w, l, d = run_side(X)
    print(f"Network as X vs perfect solver as O: {w} wins, {l} losses, {d} draws")
    w, l, d = run_side(O)
    print(f"Network as O vs perfect solver as X: {w} wins, {l} losses, {d} draws")


if __name__ == "__main__":
    print("Enumerating solver states...", flush=True)
    states, transitions = _enumerate_states()
    values, _ = _value_iteration(states, transitions)
    print(f"  {len(values)} states.", flush=True)

    all_keys = list(values.keys())
    random.shuffle(all_keys)
    split = int(len(all_keys) * 0.8)
    train_keys, test_keys = all_keys[:split], all_keys[split:]

    print("Building training examples (solver optimal-move targets, ties included)...", flush=True)
    train_examples = build_examples(train_keys)
    test_examples = build_examples(test_keys)
    print(f"  train: {len(train_examples)}, test: {len(test_examples)}", flush=True)

    net = TicTacToeNet(hidden_size=64, seed=42)

    NUM_EPOCHS = 60
    for epoch in range(NUM_EPOCHS):
        value_loss, policy_loss = train_epoch(net, train_examples)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_value_mae, test_policy_acc = evaluate(net, test_examples)
            print(f"epoch {epoch + 1:>3}/{NUM_EPOCHS}: train value_loss={value_loss:.4f} policy_loss={policy_loss:.4f} "
                  f"| held-out: value_MAE={test_value_mae:.4f} policy_top1_acc={test_policy_acc:.4f}", flush=True)

    net.save('stage1_policy_imitation_net.pkl')
    print("Saved to stage1_policy_imitation_net.pkl", flush=True)

    print("\n=== Decisive test: network plays both sides itself, no help, vs perfect solver ===", flush=True)
    decisive_test(net)
