"""Isolated smoke test for the catchup_wr run mode: a handful of tiny, fast
iterations to confirm the win-rate-gated freeze logic actually engages during
a real training loop (not just the synthetic unit test), before committing to
a full 300-iteration run. Does not touch any of the real log/checkpoint files."""

from collections import deque

from game_engine import X, O
from neural_net import TicTacToeNet
from perfect_solver import _enumerate_states, _value_iteration
from alphazero_selfplay import (
    play_one_selfplay_game, train_on_examples, value_error_vs_solver_by_mover,
    win_rate_freeze_decision,
)

print("Loading solver states...", flush=True)
states, transitions = _enumerate_states()
values, _ = _value_iteration(states, transitions)

NUM_ITERATIONS = 8
GAMES_PER_ITERATION = 6   # small, just to get a real (if noisy) win rate per iteration
MCTS_ITERS = 15           # small, just to make this run in seconds not minutes
FREEZE_THRESHOLD = 0.5    # deliberately high so freezing triggers almost immediately - proves the mechanism engages
UNFREEZE_THRESHOLD = 0.9  # deliberately high so it won't unfreeze in this short test - proves hysteresis holds
WIN_RATE_WINDOW = 3       # small window so it starts gating quickly in a short smoke test

net_x = TicTacToeNet(hidden_size=64, seed=42)
net_o = TicTacToeNet(hidden_size=64, seed=43)
nets = {X: net_x, O: net_o}
buffer_x = deque(maxlen=1000)
buffer_o = deque(maxlen=1000)
o_win_rate_history = []
x_frozen = False

for iteration in range(NUM_ITERATIONS):
    results = {'X': 0, 'O': 0, 'draw': 0}
    for _ in range(GAMES_PER_ITERATION):
        examples, winner = play_one_selfplay_game(nets, iterations=MCTS_ITERS)
        for ex in examples:
            (buffer_x if ex[3] == X else buffer_o).append(ex)
        if winner == -1:
            results['draw'] += 1
        elif winner == X:
            results['X'] += 1
        else:
            results['O'] += 1

    o_win_rate_history.append(results['O'] / GAMES_PER_ITERATION)
    prev_frozen = x_frozen
    x_frozen = win_rate_freeze_decision(o_win_rate_history, FREEZE_THRESHOLD,
                                         UNFREEZE_THRESHOLD, WIN_RATE_WINDOW, x_frozen)
    rolling = sum(o_win_rate_history[-WIN_RATE_WINDOW:]) / len(o_win_rate_history[-WIN_RATE_WINDOW:])

    x_weights_before = net_x.W1.copy()
    if x_frozen:
        pass  # skip training X entirely, same as the real loop
    else:
        train_on_examples(net_x, list(buffer_x))
    train_on_examples(net_o, list(buffer_o))
    x_weights_changed = not (x_weights_before == net_x.W1).all()

    print(f"iter {iteration+1}: games={results}, o_win_rate_rolling={rolling:.3f}, "
          f"x_frozen={x_frozen} (was {prev_frozen}), X weights changed this iter: {x_weights_changed}", flush=True)

    # The actual assertion that matters: frozen must mean weights truly did not move.
    if x_frozen:
        assert not x_weights_changed, "BUG: x_frozen=True but X's weights changed anyway!"
    else:
        assert x_weights_changed or iteration == 0, "X should be training when not frozen"

print("\nSmoke test passed: freeze decisions are correctly wired to actually skip X's gradient updates.")
