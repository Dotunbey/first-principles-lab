"""
Classic Tic-Tac-Toe (the standard 3x3 game) - no sliding phase, no piece cap.
Players alternate placing a mark on any empty cell until someone gets three
in a row, or the board fills up (draw).

Board cells are indexed:
    0 1 2
    3 4 5
    6 7 8

Deliberately kept interface-compatible with sliding-tic-tac-toe/game_engine.py
(board, current_player, phase, game_over, winner, get_valid_moves(),
make_move(), copy()) so mcts.py, q_learning.py, and the web dashboard can be
reused almost unchanged - the only real difference is get_valid_moves()/
make_move() never enter a 'sliding' phase; `phase` stays 'placement' for the
entire game, which is also what makes warm-starting a Q-table from the
sliding project's placement-phase entries meaningful (see
bootstrap_from_sliding_agent.py) - those states are byte-for-byte identical
board positions, since sliding-game placement plays out exactly like this
game for the first six plies.
"""

EMPTY, X, O = 0, 1, 2

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


class TicTacToe:
    def __init__(self):
        self.board = [EMPTY] * 9
        self.current_player = X
        self.phase = 'placement'  # always - this game never slides
        self.game_over = False
        self.winner = None  # X, O, or -1 for draw

    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i] == EMPTY]

    def make_move(self, index):
        if self.board[index] != EMPTY:
            raise ValueError("Invalid move: cell occupied")

        self.board[index] = self.current_player

        if self._check_win(index):
            return

        if EMPTY not in self.board:
            self.game_over = True
            self.winner = -1
            return

        self.current_player = O if self.current_player == X else X

    def _check_win(self, last_index):
        player = self.board[last_index]
        for line in WIN_LINES:
            if last_index in line and all(self.board[i] == player for i in line):
                self.game_over = True
                self.winner = player
                return True
        return False

    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def print_board(self):
        symbols = {EMPTY: '.', X: 'X', O: 'O'}
        for row in range(3):
            print(' '.join(symbols[self.board[row * 3 + col]] for col in range(3)))
        print(f"Player to move: {symbols[self.current_player]}")
