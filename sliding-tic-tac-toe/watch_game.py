"""
Watch a single game play out move-by-move on an actual board, instead of
reading printed text. Point `watch_game` at any two functions that each
take the current game state and return a legal move.
"""

import matplotlib.pyplot as plt

from game_engine import SlidingTicTacToe, X, O
from board_view import draw_board


def watch_game(player_X_fn, player_O_fn, pause=0.6, title="Sliding Tic-Tac-Toe"):
    """player_X_fn / player_O_fn: functions of (game) -> move."""
    game = SlidingTicTacToe()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.canvas.manager.set_window_title(title)
    plt.ion()

    move_number = 0
    draw_board(ax, game, move_number)
    plt.pause(pause)

    while not game.game_over:
        move_number += 1
        move = player_X_fn(game) if game.current_player == X else player_O_fn(game)
        game.make_move(move)
        draw_board(ax, game, move_number)
        plt.pause(pause)

    if game.winner == -1:
        result = "Draw!"
    else:
        result = f"{'X' if game.winner == X else 'O'} wins!"
    ax.set_title(result)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import random

    from mcts import mcts_search
    from q_learning import QLearningAgent

    agent = QLearningAgent()
    try:
        agent.load_model('q_learning_model.pkl')
    except FileNotFoundError:
        print("No trained Q-learning model found - run q_learning.py or training_curve.py first.")
        raise SystemExit(1)

    def q_agent_move(game):
        return agent.choose_action(game, greedy=True)

    def mcts_move(game):
        return mcts_search(game, iterations=300)

    watch_game(q_agent_move, mcts_move, title="Trained Q-agent (X) vs MCTS (O)")
