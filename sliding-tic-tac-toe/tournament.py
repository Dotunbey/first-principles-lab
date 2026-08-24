"""
Round-robin tournament across every trained neural-net variant from the
AlphaZero-style build (research/alphazero_deep_dive/, research/VARIANT_REGISTRY.md),
plus the project's non-learned baselines (perfect solver, plain MCTS, random).

Every matchup is played GREEDY (no Dirichlet noise, no temperature sampling -
neural_mcts_search(add_noise=False)), which makes each matchup fully
deterministic given a fixed iteration count - so unlike self-play training,
there's no point replaying the same ordered pairing twice, one game per
ordered (X-side, O-side) pair is the actual, repeatable result.

This directly extends the decisive test used throughout
research/alphazero_deep_dive/05_results_summary.md (every variant vs the
perfect solver, both sides) to variant-vs-variant and variant-vs-baseline
matchups, which hadn't been tested before - only each variant in isolation
against the solver.
"""

import csv
import time

from game_engine import SlidingTicTacToe, X, O
from neural_net import TicTacToeNet
from neural_mcts import neural_mcts_search
from mcts import mcts_search
from perfect_solver import best_move as perfect_solver_move
import random

MCTS_ITERATIONS = 150     # matches this project's typical "live" MCTS strength
PLAIN_MCTS_ITERATIONS = 200


def load_net(path, seed=42):
    net = TicTacToeNet(hidden_size=64, seed=seed)
    net.load(path)
    return net


# Each entry: name -> a move-choosing function(game) -> move, using that
# competitor's strongest available play (greedy, no exploration randomness).
def make_network_player(net):
    def player(game):
        move, _ = neural_mcts_search(game, net, iterations=MCTS_ITERATIONS, add_noise=False)
        return move
    return player


def make_baseline_players():
    def solver_player(game):
        return perfect_solver_move(game)

    def mcts_player(game):
        return mcts_search(game, iterations=PLAIN_MCTS_ITERATIONS)

    def random_player(game):
        return random.choice(game.get_valid_moves())

    return {'perfect_solver': solver_player, 'plain_mcts': mcts_player, 'random': random_player}


def build_competitors():
    competitors = {}

    single_net_variants = {
        'unbalanced': 'selfplay_trained_net.pkl',
        'balanced': 'selfplay_trained_net_balanced.pkl',
        'teacher': 'selfplay_trained_net_teacher.pkl',
        'exploration': 'selfplay_trained_net_exploration.pkl',
        'o_compute': 'selfplay_trained_net_o_compute.pkl',
        'imitate': 'selfplay_trained_net_imitate.pkl',
    }
    for name, path in single_net_variants.items():
        net = load_net(path)
        competitors[name] = {'X': make_network_player(net), 'O': make_network_player(net)}

    dual_net_variants = {
        'twonet': ('selfplay_trained_net_twonet_X.pkl', 'selfplay_trained_net_twonet_O.pkl'),
        'catchup': ('selfplay_trained_net_catchup_X.pkl', 'selfplay_trained_net_catchup_O.pkl'),
        'catchup_wr': ('selfplay_trained_net_catchup_wr_X.pkl', 'selfplay_trained_net_catchup_wr_O.pkl'),
    }
    for name, (x_path, o_path) in dual_net_variants.items():
        net_x = load_net(x_path, seed=42)
        net_o = load_net(o_path, seed=43)
        competitors[name] = {'X': make_network_player(net_x), 'O': make_network_player(net_o)}

    baselines = make_baseline_players()
    for name, fn in baselines.items():
        competitors[name] = {'X': fn, 'O': fn}

    return competitors


def play_game(x_player_fn, o_player_fn):
    game = SlidingTicTacToe()
    while not game.game_over:
        fn = x_player_fn if game.current_player == X else o_player_fn
        move = fn(game)
        game.make_move(move)
    return game.winner  # X, O, or -1 for draw


def run_tournament(competitors):
    names = list(competitors.keys())
    results = []  # (x_name, o_name, winner)
    total = len(names) * (len(names) - 1)
    done = 0
    start = time.time()

    for x_name in names:
        for o_name in names:
            if x_name == o_name:
                continue
            winner = play_game(competitors[x_name]['X'], competitors[o_name]['O'])
            outcome = 'draw' if winner == -1 else ('X' if winner == X else 'O')
            results.append((x_name, o_name, outcome))
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} games played ({time.time()-start:.1f}s elapsed)", flush=True)

    return results


def summarize(results, names):
    """Per-competitor record: wins/losses/draws pooled over BOTH the games
    where it played X and where it played O - the single number that answers
    'how good is this variant overall,' as opposed to the per-side numbers
    the decisive test already established (every variant strong as X, weak
    as O vs the solver specifically)."""
    record = {name: {'wins': 0, 'losses': 0, 'draws': 0} for name in names}
    for x_name, o_name, outcome in results:
        if outcome == 'draw':
            record[x_name]['draws'] += 1
            record[o_name]['draws'] += 1
        elif outcome == 'X':
            record[x_name]['wins'] += 1
            record[o_name]['losses'] += 1
        else:
            record[o_name]['wins'] += 1
            record[x_name]['losses'] += 1
    return record


if __name__ == "__main__":
    print("Loading all competitors (9 network variants + 3 baselines)...", flush=True)
    competitors = build_competitors()
    names = list(competitors.keys())
    print(f"Loaded {len(names)} competitors: {names}", flush=True)

    print("Running full round robin (every ordered pair plays once, deterministic/greedy)...", flush=True)
    results = run_tournament(competitors)

    with open('tournament_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_player', 'o_player', 'outcome'])
        writer.writerows(results)

    record = summarize(results, names)
    print()
    print(f"{'Name':<16} {'Wins':>5} {'Losses':>7} {'Draws':>6}")
    for name in sorted(names, key=lambda n: -record[n]['wins']):
        r = record[name]
        print(f"{name:<16} {r['wins']:>5} {r['losses']:>7} {r['draws']:>6}")

    print()
    print("Full per-game results saved to tournament_results.csv")
