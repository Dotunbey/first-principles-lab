"""
Monte Carlo Tree Search for Sliding Tic-Tac-Toe.

No training, no Q-table - at decision time, run many simulated games from the
current position and pick the move that led to the most wins.

Four steps, repeated `iterations` times per real move:
  1. Selection    - walk down the tree using UCB1 until a node with untried
                     moves (or a terminal state) is reached
  2. Expansion     - add one new child for an untried move
  3. Simulation    - play the rest of that game out with random moves
  4. Backpropagation - push the result back up every node on the path taken
"""

import math
import random

from game_engine import SlidingTicTacToe

EXPLORATION_CONSTANT = 1.41  # ~sqrt(2), standard default


class TreeNode:
    def __init__(self, game_state, parent=None, move=None, player_just_moved=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # the move that led from parent to this node
        self.player_just_moved = player_just_moved
        self.children = []
        self.untried_moves = [] if game_state.game_over else game_state.get_valid_moves()
        self.visits = 0
        self.wins = 0.0

    def ucb1_score(self, parent_visits):
        exploitation = self.wins / self.visits
        exploration = EXPLORATION_CONSTANT * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        """Pick the child with the highest UCB1 score."""
        return max(self.children, key=lambda child: child.ucb1_score(self.visits))

    def expand(self):
        """Turn one untried move into a new child node."""
        move = self.untried_moves.pop()
        player_just_moved = self.game_state.current_player

        next_state = self.game_state.copy()
        next_state.make_move(move)

        child = TreeNode(next_state, parent=self, move=move, player_just_moved=player_just_moved)
        self.children.append(child)
        return child

    def backpropagate(self, winner):
        node = self
        while node is not None:
            node.visits += 1
            if winner == -1:
                node.wins += 0.5  # draw
            elif winner == node.player_just_moved:
                node.wins += 1
            node = node.parent


def heuristic_rollout_move(game):
    """One-ply-lookahead move choice for the rollout phase: take an
    immediate win if one exists, otherwise avoid any move that hands the
    opponent an immediate win on their next turn. This is what turns a
    'purely random' simulated game into a 'competent' one - it doesn't
    change WHAT MCTS is doing (still Monte Carlo simulation), just how
    informative each simulated game's result actually is."""
    valid_moves = game.get_valid_moves()
    mover = game.current_player

    for move in valid_moves:
        trial = game.copy()
        trial.make_move(move)
        if trial.game_over and trial.winner == mover:
            return move  # take a free win instead of ever missing it

    safe_moves = []
    for move in valid_moves:
        trial = game.copy()
        trial.make_move(move)
        if trial.game_over:
            safe_moves.append(move)  # draw or already handled win case above
            continue
        opponent_can_win_next = False
        for reply in trial.get_valid_moves():
            reply_trial = trial.copy()
            reply_trial.make_move(reply)
            if reply_trial.game_over and reply_trial.winner == trial.current_player:
                opponent_can_win_next = True
                break
        if not opponent_can_win_next:
            safe_moves.append(move)

    candidates = safe_moves if safe_moves else valid_moves
    return random.choice(candidates)


def rollout(game_state):
    """Play out a copy of the game using the heuristic move choice above
    until it ends - a 'smart-ish' simulation instead of a purely random one."""
    game = game_state.copy()
    while not game.game_over:
        move = heuristic_rollout_move(game)
        game.make_move(move)
    return game.winner


def mcts_search(root_state, iterations=1000):
    """Run MCTS from root_state and return the move with the most visits."""
    root = TreeNode(root_state.copy())

    for _ in range(iterations):
        node = root

        # 1. Selection - descend while fully expanded and non-terminal
        while not node.untried_moves and node.children and not node.game_state.game_over:
            node = node.select_child()

        # 2. Expansion
        if node.untried_moves and not node.game_state.game_over:
            node = node.expand()

        # 3. Simulation
        winner = rollout(node.game_state)

        # 4. Backpropagation
        node.backpropagate(winner)

    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move


def demonstrate_mcts_vs_random(num_games=20, iterations=200):
    """Sanity check: MCTS (playing X) should beat a random player (playing O)
    the large majority of the time."""
    results = {'X': 0, 'O': 0, 'draw': 0}

    for game_num in range(num_games):
        game = SlidingTicTacToe()
        while not game.game_over:
            if game.current_player == 1:  # X = MCTS
                move = mcts_search(game, iterations=iterations)
            else:  # O = random
                move = random.choice(game.get_valid_moves())
            game.make_move(move)

        if game.winner == 1:
            results['X'] += 1
        elif game.winner == 2:
            results['O'] += 1
        else:
            results['draw'] += 1

        print(f"Game {game_num + 1}/{num_games} complete - winner: "
              f"{'X (MCTS)' if game.winner == 1 else 'O (random)' if game.winner == 2 else 'draw'}")

    print()
    print(f"Results over {num_games} games: {results}")
    print(f"MCTS win rate: {results['X'] / num_games * 100:.1f}%")


if __name__ == "__main__":
    demonstrate_mcts_vs_random()
