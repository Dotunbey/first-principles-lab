"""
Replay any single game from game_log.json, move by move, on an actual board.

Usage:
    python replay_game.py little           # replays the shortest logged game
    python replay_game.py longest          # replays the longest logged game
    python replay_game.py 42                # replays game number 42
"""

import json
import sys

import matplotlib.pyplot as plt

from game_engine import SlidingTicTacToe
from board_view import draw_board


def load_game_log(path='game_log.json'):
    with open(path) as f:
        return json.load(f)


def _to_move(raw_move):
    """JSON turns tuples into lists - convert (from, to) slide moves back."""
    return tuple(raw_move) if isinstance(raw_move, list) else raw_move


def replay(entry, pause=0.6):
    game = SlidingTicTacToe()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.canvas.manager.set_window_title(
        f"Game {entry['game_num']} | Q played {entry['q_plays']} | outcome: {entry['outcome']}")
    plt.ion()

    draw_board(ax, game, 0)
    plt.pause(pause)

    for i, raw_move in enumerate(entry['moves']):
        move = _to_move(raw_move)
        game.make_move(move)
        draw_board(ax, game, i + 1)
        plt.pause(pause)

    if game.winner == -1:
        result = "Draw!"
    else:
        result = f"{'X' if game.winner == 1 else 'O'} wins!"
    ax.set_title(f"Game {entry['game_num']} - {result} (Q played {entry['q_plays']})")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    log = load_game_log()

    if len(sys.argv) < 2:
        print(f"{len(log)} games logged. Usage: python replay_game.py <game_number|shortest|longest>")
        raise SystemExit(1)

    selector = sys.argv[1]
    if selector == 'shortest':
        entry = min(log, key=lambda g: g['move_count'])
    elif selector == 'longest':
        entry = max(log, key=lambda g: g['move_count'])
    else:
        game_num = int(selector)
        entry = next(g for g in log if g['game_num'] == game_num)

    replay(entry)
