"""
Sliding Tic-Tac-Toe (Three Men's Morris) Game Engine

Board cells are indexed:
    0 1 2
    3 4 5
    6 7 8

Two phases:
  - placement: each player places 3 pieces onto empty cells (like normal
    tic-tac-toe, but stops after 3 each)
  - sliding: once both players have placed all 3 pieces, each turn a player
    slides one of their own pieces into an adjacent empty cell

Adjacency follows the 8 win-lines (3 rows, 3 columns, 2 diagonals) - two
cells are adjacent if they sit next to each other on one of those lines.
Corners and edge-midpoints have 3 neighbors each; the center has 8 (it
touches every line).
"""

EMPTY, X, O = 0, 1, 2

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]

ADJACENCY = {
    0: [1, 3, 4],
    1: [0, 2, 4],
    2: [1, 4, 5],
    3: [0, 4, 6],
    4: [0, 1, 2, 3, 5, 6, 7, 8],
    5: [2, 4, 8],
    6: [3, 4, 7],
    7: [4, 6, 8],
    8: [4, 5, 7],
}

PIECES_PER_PLAYER = 3
MAX_SLIDE_MOVES = 40  # move-limit draw, since sliding can otherwise go on forever


class SlidingTicTacToe:
    def __init__(self):
        self.board = [EMPTY] * 9
        self.current_player = X
        self.phase = 'placement'
        self.pieces_placed = {X: 0, O: 0}
        self.slide_move_count = 0
        self.position_counts = {}  # (board tuple, player) -> times seen, for threefold repetition
        self.game_over = False
        self.winner = None  # X, O, or -1 for draw

    def get_valid_moves(self):
        """Placement phase: list of cell indices. Sliding phase: list of (from, to) tuples."""
        if self.phase == 'placement':
            return [i for i in range(9) if self.board[i] == EMPTY]

        moves = []
        for i in range(9):
            if self.board[i] != self.current_player:
                continue
            for j in ADJACENCY[i]:
                if self.board[j] == EMPTY:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        if self.phase == 'placement':
            self._make_placement_move(move)
        else:
            self._make_slide_move(move)

    def _make_placement_move(self, index):
        if self.board[index] != EMPTY:
            raise ValueError("Invalid move: cell occupied")

        self.board[index] = self.current_player
        self.pieces_placed[self.current_player] += 1

        if self._check_win(index):
            return

        if self.pieces_placed[X] == PIECES_PER_PLAYER and self.pieces_placed[O] == PIECES_PER_PLAYER:
            self.phase = 'sliding'

        self.current_player = O if self.current_player == X else X

    def _make_slide_move(self, move):
        from_idx, to_idx = move
        if self.board[from_idx] != self.current_player:
            raise ValueError("Invalid move: not your piece")
        if to_idx not in ADJACENCY[from_idx] or self.board[to_idx] != EMPTY:
            raise ValueError("Invalid move: illegal slide")

        self.board[to_idx] = self.current_player
        self.board[from_idx] = EMPTY
        self.slide_move_count += 1

        if self._check_win(to_idx):
            return

        if self._check_repetition_draw():
            return

        if self.slide_move_count >= MAX_SLIDE_MOVES:
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

    def _check_repetition_draw(self):
        next_player = O if self.current_player == X else X
        key = (tuple(self.board), next_player)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        if self.position_counts[key] >= 3:
            self.game_over = True
            self.winner = -1
            return True
        return False

    def copy(self):
        new_game = SlidingTicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.phase = self.phase
        new_game.pieces_placed = self.pieces_placed.copy()
        new_game.slide_move_count = self.slide_move_count
        new_game.position_counts = self.position_counts.copy()
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def display(self):
        symbols = {EMPTY: '.', X: 'X', O: 'O'}
        for row in range(3):
            print(' '.join(symbols[self.board[row * 3 + c]] for c in range(3)))
        print(f"Phase: {self.phase} | Player to move: {symbols[self.current_player]}")
        if self.phase == 'sliding':
            print(f"Slide moves so far: {self.slide_move_count}/{MAX_SLIDE_MOVES}")


def main():
    game = SlidingTicTacToe()
    print("Welcome to Sliding Tic-Tac-Toe!")
    print()

    while not game.game_over:
        game.display()
        valid_moves = game.get_valid_moves()
        print("Valid moves:", valid_moves)

        try:
            if game.phase == 'placement':
                move = int(input("Choose a cell (0-8): "))
            else:
                raw = input("Choose a move as 'from,to': ")
                move = tuple(int(x) for x in raw.split(','))
            game.make_move(move)
        except (ValueError, IndexError):
            print("Invalid move! Try again.")
            continue
        print()

    game.display()
    if game.winner == -1:
        print("Game ended in a draw!")
    else:
        symbols = {X: 'X', O: 'O'}
        print(f"Player {symbols[game.winner]} wins!")


if __name__ == "__main__":
    main()
