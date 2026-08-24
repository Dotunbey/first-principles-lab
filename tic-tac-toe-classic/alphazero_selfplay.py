"""
The full AlphaZero-style closed loop for classic tic-tac-toe - same design
as ../sliding-tic-tac-toe/alphazero_selfplay.py's base ("unbalanced") run:
network-guided self-play games generate their own training data (visit-count
policy target + real game outcome), which then trains the network that
guides the next round of self-play.

Unlike the sliding project, this game has NO known X/O asymmetry to fight -
classic tic-tac-toe is a proven draw with optimal play on BOTH sides (unlike
the sliding game, where X has a real forced win) - so none of the sliding
project's asymmetry-fix variants (teacher, twonet, catchup, imitate, ...)
apply here. This is the base build only; if a similar asymmetry shows up
empirically anyway (it might, from self-play dynamics alone, the same way
the sliding game's X advantage compounded through a shared network), that
would be a genuinely new finding worth its own investigation.
"""

import random
from collections import deque

import numpy as np

from game_engine import TicTacToe, X, O
from neural_net import TicTacToeNet, encode_state, encode_action, POLICY_SIZE
from neural_mcts import neural_mcts_search, visit_count_policy
from perfect_solver import negamax_value


def play_one_selfplay_game(net, iterations=100, temperature_moves=4, noise_frac=0.25):
    """Returns a list of (input_vector, policy_target, value_target, mover)
    for every position visited, plus the game's winner."""
    game = TicTacToe()
    examples = []
    move_number = 0

    while not game.game_over:
        move, root = neural_mcts_search(game, net, iterations=iterations, add_noise=True, noise_frac=noise_frac)

        temperature = 1.0 if move_number < temperature_moves else 0.1
        policy_target = visit_count_policy(root, temperature=temperature)

        x = encode_state(game.board, game.phase, game.current_player)
        examples.append((x, policy_target, game.current_player))

        if temperature > 0.5:
            legal_moves = list(root.children.keys())
            legal_indices = [encode_action(m) for m in legal_moves]
            probs = np.array([policy_target[i] for i in legal_indices])
            probs = probs / probs.sum()
            move = legal_moves[np.random.choice(len(legal_moves), p=probs)]

        game.make_move(move)
        move_number += 1

    winner = game.winner
    training_examples = []
    for x, policy_target, mover in examples:
        if winner == -1:
            value_target = 0.0
        else:
            value_target = 1.0 if winner == mover else -1.0
        training_examples.append((x, policy_target, value_target, mover))

    return training_examples, winner


def value_error_vs_solver(net, sample_states, sample_size=300):
    """Mean |net's value - solver's exact value| over a random sample of real
    (non-terminal) states - this project's zero-noise 'training loss' metric."""
    keys = random.sample(list(sample_states.keys()), min(sample_size, len(sample_states)))
    total_error = 0.0
    for board, phase, player in keys:
        game = TicTacToe()
        game.board = list(board)
        game.phase = phase
        game.current_player = player
        true_value = negamax_value(game)
        pred_value, _ = net.predict(board, phase, player)
        total_error += abs(pred_value - true_value)
    return total_error / len(keys)


def value_error_vs_solver_by_mover(net, sample_states, sample_size=300):
    """Same, split by whose turn it is at each sampled state."""
    keys = random.sample(list(sample_states.keys()), min(sample_size, len(sample_states)))
    errors = {X: [], O: []}
    for board, phase, player in keys:
        game = TicTacToe()
        game.board = list(board)
        game.phase = phase
        game.current_player = player
        true_value = negamax_value(game)
        pred_value, _ = net.predict(board, phase, player)
        errors[player].append(abs(pred_value - true_value))
    err_x = sum(errors[X]) / len(errors[X]) if errors[X] else None
    err_o = sum(errors[O]) / len(errors[O]) if errors[O] else None
    return err_x, err_o


def balance_by_mover(examples):
    """X moves first every game and (since classic tic-tac-toe plays to a
    full board rather than switching phases) gets 5 placements per game to
    O's 4 - so X-to-move examples are naturally somewhat over-represented.
    Oversample the minority side so every training batch sees a roughly
    equal split of both roles - same fix used throughout the sliding
    project for the same underlying reason (more of one role's examples
    than the other's)."""
    by_mover = {X: [], O: []}
    for example in examples:
        by_mover[example[3]].append(example)

    if not by_mover[X] or not by_mover[O]:
        return examples

    majority_side = X if len(by_mover[X]) >= len(by_mover[O]) else O
    minority_side = O if majority_side == X else X
    target_count = len(by_mover[majority_side])

    oversampled_minority = random.choices(by_mover[minority_side], k=target_count)
    return by_mover[majority_side] + oversampled_minority


def train_on_examples(net, examples, batch_size=64, learning_rate=0.001):
    examples = balance_by_mover(examples)
    random.shuffle(examples)
    total_value_loss = 0.0
    total_policy_loss = 0.0
    n_batches = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        x_batch = np.array([e[0] for e in batch])
        policy_batch = np.array([e[1] for e in batch])
        value_batch = np.array([[e[2]] for e in batch])
        legal_mask = (policy_batch > 0).astype(float)
        value_loss, policy_loss = net.train_step(x_batch, value_batch, policy_batch, legal_mask, learning_rate)
        total_value_loss += value_loss
        total_policy_loss += policy_loss
        n_batches += 1

    return total_value_loss / n_batches, total_policy_loss / n_batches


if __name__ == "__main__":
    import csv
    import os
    import sys
    import time

    from perfect_solver import enumerate_all_states

    print("Enumerating all non-terminal states for value-error benchmarking...", flush=True)
    values = enumerate_all_states()
    print(f"  {len(values)} states enumerated.", flush=True)

    NUM_ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    RUN_NAME = sys.argv[2] if len(sys.argv) > 2 else 'base'
    TEMPERATURE_MOVES = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    NOISE_FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    GAMES_PER_ITERATION = 20
    MCTS_ITERS = 100
    CHECKPOINT_EVERY = 5
    REPLAY_BUFFER_SIZE = 4000  # smaller than the sliding project's 8000 - this game is much smaller

    log_path = f'selfplay_training_log_{RUN_NAME}.csv'
    net_save_path = f'selfplay_trained_net_{RUN_NAME}.pkl'

    print(f"Run '{RUN_NAME}': iterations={NUM_ITERATIONS}, temperature_moves={TEMPERATURE_MOVES}, "
          f"noise_frac={NOISE_FRAC}", flush=True)

    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(['iteration', 'wins_X', 'wins_O', 'draws',
                              'buffer_size', 'value_loss', 'policy_loss',
                              'value_error_vs_solver', 'value_error_X', 'value_error_O',
                              'elapsed_sec'])

    net = TicTacToeNet(hidden_size=32, seed=42)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    start_time = time.time()

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.time()
        results = {'X': 0, 'O': 0, 'draw': 0}

        for _ in range(GAMES_PER_ITERATION):
            examples, winner = play_one_selfplay_game(net, iterations=MCTS_ITERS,
                                                       temperature_moves=TEMPERATURE_MOVES, noise_frac=NOISE_FRAC)
            replay_buffer.extend(examples)
            if winner == -1:
                results['draw'] += 1
            elif winner == X:
                results['X'] += 1
            else:
                results['O'] += 1

        value_loss, policy_loss = train_on_examples(net, list(replay_buffer))
        err = value_error_vs_solver(net, values)
        err_x, err_o = value_error_vs_solver_by_mover(net, values)
        elapsed = time.time() - iter_start

        print(f"iter {iteration + 1:>4}/{NUM_ITERATIONS}: games={results}, buffer={len(replay_buffer)}, "
              f"value_loss={value_loss:.4f}, policy_loss={policy_loss:.4f}, "
              f"value_error_vs_solver={err:.4f} (X={err_x:.4f}, O={err_o:.4f}), elapsed={elapsed:.1f}s", flush=True)

        log_writer.writerow([iteration + 1, results['X'], results['O'], results['draw'],
                              len(replay_buffer), value_loss, policy_loss, err, err_x, err_o, elapsed])
        log_file.flush()

        if (iteration + 1) % CHECKPOINT_EVERY == 0:
            net.save(net_save_path)

    net.save(net_save_path)
    log_file.close()
    print(f"Done. Total time: {time.time() - start_time:.1f}s. Saved to {net_save_path}", flush=True)
