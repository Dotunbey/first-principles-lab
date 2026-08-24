"""
Exact solver for classic tic-tac-toe via memoized minimax.

Unlike the sliding variant (../sliding-tic-tac-toe/perfect_solver.py), this
game has NO cycles - every move fills an empty cell, so the state space is a
strict DAG (at most 9 plies deep) and plain recursive minimax with a
transposition cache is completely sound here; the sliding game's more
involved value-iteration-over-the-full-graph approach isn't needed. Standard
result: with optimal play on both sides, classic tic-tac-toe is a draw.
"""

from functools import lru_cache

from game_engine import TicTacToe, X, O


def _other(player):
    return O if player == X else X


@lru_cache(maxsize=None)
def _negamax(board, current_player):
    """Returns the exact value of this position from current_player's
    perspective: +1 forced win, -1 forced loss, 0 with best play (draw)."""
    game = TicTacToe()
    game.board = list(board)
    game.current_player = current_player

    # Check terminal purely from the board (no move has been "just made" at
    # the root of this call, so check every possible completed line):
    for line in ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
                 (2, 5, 8), (0, 4, 8), (2, 4, 6)):
        a, b, c = line
        if board[a] != 0 and board[a] == board[b] == board[c]:
            winner = board[a]
            return 1.0 if winner == current_player else -1.0

    valid_moves = [i for i in range(9) if board[i] == 0]
    if not valid_moves:
        return 0.0  # draw - board full, no winner

    best = -2.0
    for move in valid_moves:
        next_board = list(board)
        next_board[move] = current_player
        value = -_negamax(tuple(next_board), _other(current_player))
        if value > best:
            best = value
    return best


def negamax_value(game):
    """Exact value of `game`'s current position, from whoever's turn it is
    right now. +1 = forced win, -1 = forced loss, 0 = draw with best play."""
    if game.game_over:
        if game.winner == -1:
            return 0.0
        return 1.0 if game.winner == game.current_player else -1.0
    return _negamax(tuple(game.board), game.current_player)


def best_move(game):
    """The provably optimal move from this position (ties broken by move
    order - any tied move is equally optimal)."""
    valid_moves = game.get_valid_moves()
    best = None
    best_value = -2.0
    for move in valid_moves:
        trial = game.copy()
        trial.make_move(move)
        if trial.game_over:
            value = 0.0 if trial.winner == -1 else (1.0 if trial.winner == game.current_player else -1.0)
        else:
            value = -_negamax(tuple(trial.board), trial.current_player)
        if value > best_value:
            best_value = value
            best = move
    return best


def enumerate_all_states():
    """Every reachable NON-TERMINAL (board, phase, player) state - i.e. every
    position an agent might actually be asked to evaluate or choose a move
    from - mapped to its exact solved value. The classic-tic-tac-toe
    equivalent of the sliding project's _enumerate_states()+_value_iteration()
    pair, but much simpler: no cycles here, so a single BFS plus the
    already-memoized _negamax is sufficient, no value-iteration sweep needed.

    Terminal (game-over) positions are deliberately excluded from the
    result, for two reasons: (1) nothing in this project ever asks a network
    or Q-table to evaluate an already-finished game - MCTS's _terminal_value
    computes those directly from the game result, bypassing the network
    entirely (see neural_mcts.py) - so a terminal state is never actually
    sampled/queried in practice; (2) game_engine.py's `current_player` field
    is NOT reliable at a terminal node - make_move() returns early on a win,
    before flipping whose turn it is (the same bug documented for the
    sliding game's engine, research/../04_bugs_lessons_and_limitations.md) -
    so a terminal state's "player" component of its key would be ambiguous
    anyway. Skipping terminal states sidesteps this by construction rather
    than by remembering not to trust the field."""
    from collections import deque
    from game_engine import TicTacToe

    start = TicTacToe()
    start_key = (tuple(start.board), start.phase, start.current_player)
    values = {start_key: negamax_value(start)}
    queue = deque([start])

    while queue:
        game = queue.popleft()
        for move in game.get_valid_moves():
            child = game.copy()
            child.make_move(move)
            if child.game_over:
                continue  # terminal - not a state anything is ever asked to evaluate
            key = (tuple(child.board), child.phase, child.current_player)
            if key not in values:
                values[key] = negamax_value(child)
                queue.append(child)

    return values


if __name__ == "__main__":
    game = TicTacToe()
    value = negamax_value(game)
    print(f"Value of the empty board (X to move): {value}  (expected: 0.0 - a draw with perfect play)")

    move = best_move(game)
    print(f"Solver's opening move: {move} (center=4 or a corner are both optimal)")
