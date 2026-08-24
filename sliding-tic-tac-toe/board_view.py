"""Shared matplotlib board renderer, used by watch_game.py (live games) and
replay_game.py (replaying a logged game from training)."""

from game_engine import X, O


def draw_board(ax, game, move_number, title_override=None):
    ax.clear()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(1, 3):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)

    symbols = {X: 'X', O: 'O'}
    colors = {X: 'tab:blue', O: 'tab:red'}
    for idx in range(9):
        row, col = divmod(idx, 3)
        mark = game.board[idx]
        if mark == 0:
            continue
        cx, cy = col + 0.5, 2 - row + 0.5
        ax.text(cx, cy, symbols[mark], ha='center', va='center',
                fontsize=48, color=colors[mark], fontweight='bold')

    if title_override:
        ax.set_title(title_override)
    else:
        turn_symbol = symbols[game.current_player]
        ax.set_title(f"Move {move_number} | Phase: {game.phase} | Next to move: {turn_symbol}")
