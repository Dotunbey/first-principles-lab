"""
Warm-start a classic-tic-tac-toe Q-table from an already-trained
sliding-tic-tac-toe agent's session file, instead of starting from scratch.

RESULT (see research/warm_start_experiment.md for the full writeup): this
transfer was TESTED HEAD-TO-HEAD against a fresh agent under identical
hyperparameters, and it LOST - not just failed to help, but started out
actively worse than a blank table (mean |inherited Q-value - classic ground
truth| = 0.569 at game 0, versus the fresh agent's 0.428 at game 25) and
was still behind nearly 8,000 games later. Kept in the repo as a documented
negative result and because the bootstrap mechanics themselves are still
useful (e.g. for a future experiment with a differently-labeled state hash);
DO NOT use this as the default way to initialize a classic agent.

The reasoning that made this seem sound - and where it actually breaks:
the board positions ARE byte-identical (same board representation, same
win-lines, same turn order, same state-hash format), since both games play
out identically for their first six plies. The flaw is in what a
position's Q-VALUE actually encodes: it's not just "whose marks are where,"
it's "the expected outcome given what happens next." In the sliding game,
that means "assuming this eventually transitions to sliding-phase
maneuvering." In classic tic-tac-toe, it means "assuming direct placement
continues until the board fills." Those are different games from that
point forward, so the same board can have OPPOSITE correct values (one
verified case: the sliding-trained table rated a position 0.86, near-certain
win; the classic solver says -1.00, certain loss). The board being
identical was a coincidence of the opening; the strategy diverges
immediately after, and the inherited Q-values baked in sliding-game-specific
assumptions that classic tic-tac-toe training then had to spend real games
unlearning before making net-new progress.

Usage:
  python bootstrap_from_sliding_agent.py [source_pkl] [dest_pkl]
Defaults: source = ../sliding-tic-tac-toe/web_dashboard/live_session.pkl
          dest   = web_dashboard/live_session.pkl
"""

import os
import pickle
import sys

DEFAULT_CONFIG = {
    'opponent': 'mcts',
    'reward_shaping': 0,
    'td0': 0,
    'draw_reward': 0.0,   # classic tic-tac-toe draws are the EXPECTED outcome
                            # with optimal play (unlike the sliding game, where
                            # O's best is also a draw but for a different
                            # reason) - no need to penalize drawing here.
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'epsilon_start': 0.4,
    'epsilon_end': 0.05,
    'epsilon_decay_games': 300,
    'mcts_iterations': 150,
    'games_per_run': 50,
    'move_delay': 0.5,
    'update_delay': 0.6,
    'checkpoint_interval': 25,
    'checkpoint_eval_games': 10,
    'checkpoint_mcts_iterations': 80,
}


def bootstrap(source_pkl, dest_pkl, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.4):
    with open(source_pkl, 'rb') as f:
        source = pickle.load(f)

    q_table = {}
    kept = skipped = 0
    for state_hash, actions in source['q_table'].items():
        # state_hash format: f"{tuple(board)}_{phase}_{player}" - rsplit by
        # '_' from the right, since the board tuple itself may contain
        # commas but never the phase string 'placement'/'sliding'.
        _, phase, _ = state_hash.rsplit('_', 2)
        if phase != 'placement':
            skipped += 1
            continue
        q_table[state_hash] = dict(actions)
        kept += 1

    data = {
        'q_table': q_table,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'exploration_rate': exploration_rate,
        'games_played': 0,   # zero REAL classic games played - these are inherited values only
        'wins': {1: 0, 2: 0},
        'draws': 0,
        'games_run_total': 0,
        'config': dict(DEFAULT_CONFIG),
        'outcome_log': [],
        'checkpoint_history': [],
    }

    os.makedirs(os.path.dirname(dest_pkl) or '.', exist_ok=True)
    with open(dest_pkl, 'wb') as f:
        pickle.dump(data, f)

    print(f"Read {len(source['q_table'])} states from {source_pkl}")
    print(f"  kept {kept} placement-phase states (valid warm start for classic rules)")
    print(f"  discarded {skipped} sliding-phase states (no classic-game equivalent)")
    print(f"Saved warm-started session to {dest_pkl}")
    print(f"games_played reset to 0 - this agent hasn't played a single REAL classic "
          f"game yet, it has only inherited opening-move value estimates.")


if __name__ == "__main__":
    source_pkl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        '..', 'sliding-tic-tac-toe', 'web_dashboard', 'live_session.pkl')
    dest_pkl = sys.argv[2] if len(sys.argv) > 2 else os.path.join('web_dashboard', 'live_session.pkl')
    bootstrap(source_pkl, dest_pkl)
