"""Decisive playable-defense test: load a trained network (or a {X: net, O: net}
pair for the two-network variants) and have it play BOTH sides, fully on its own
(no solver help, no exploration noise, greedy best-visited move), against the
perfect solver, 20 games each way. This is the test that distinguishes actual
playable skill from value-calibration numbers alone - see
research/alphazero_deep_dive/05_results_summary.md."""

import sys

from game_engine import SlidingTicTacToe, X, O
from neural_net import TicTacToeNet
from neural_mcts import neural_mcts_search
from perfect_solver import best_move as perfect_solver_move


def play_game(net_side, solver_side, iterations=100):
    game = SlidingTicTacToe()
    while not game.game_over:
        if game.current_player == solver_side:
            move = perfect_solver_move(game)
        else:
            move, _ = neural_mcts_search(game, net_side, iterations=iterations, add_noise=False)
        game.make_move(move)
    return game.winner


def run_side(net, network_plays, iterations=100, games=20):
    wins = losses = draws = 0
    solver_side = O if network_plays == X else X
    for _ in range(games):
        winner = play_game(net, solver_side, iterations=iterations)
        if winner == network_plays:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1
    return wins, losses, draws


def decisive_test(net_x, net_o, label, games=20, iterations=100):
    print(f"=== Decisive test: {label} ===")
    w, l, d = run_side(net_x, X, iterations=iterations, games=games)
    print(f"Network playing X vs perfect solver playing O: {w} wins, {l} losses, {d} draws")
    w, l, d = run_side(net_o, O, iterations=iterations, games=games)
    print(f"Network playing O vs perfect solver playing X: {w} wins, {l} losses, {d} draws")
    print()


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "exploration"

    if variant == "exploration":
        net = TicTacToeNet(hidden_size=64, seed=42)
        net.load("selfplay_trained_net_exploration.pkl")
        decisive_test(net, net, "exploration (shared net)")

    elif variant == "o_compute":
        net = TicTacToeNet(hidden_size=64, seed=42)
        net.load("selfplay_trained_net_o_compute.pkl")
        decisive_test(net, net, "o_compute (shared net)")

    elif variant == "imitate":
        net = TicTacToeNet(hidden_size=64, seed=42)
        net.load("selfplay_trained_net_imitate.pkl")
        decisive_test(net, net, "imitate (shared net, policy-imitation fix)")

    elif variant == "twonet":
        net_x = TicTacToeNet(hidden_size=64, seed=42)
        net_x.load("selfplay_trained_net_twonet_X.pkl")
        net_o = TicTacToeNet(hidden_size=64, seed=43)
        net_o.load("selfplay_trained_net_twonet_O.pkl")
        decisive_test(net_x, net_o, "twonet (separate nets)")

    elif variant == "catchup":
        net_x = TicTacToeNet(hidden_size=64, seed=42)
        net_x.load("selfplay_trained_net_catchup_X.pkl")
        net_o = TicTacToeNet(hidden_size=64, seed=43)
        net_o.load("selfplay_trained_net_catchup_O.pkl")
        decisive_test(net_x, net_o, "catchup (separate nets, gated)")

    else:
        raise ValueError(f"unknown variant {variant}")
