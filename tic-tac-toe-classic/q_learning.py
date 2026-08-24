"""
Tabular Q-learning for classic Tic-Tac-Toe.

Same agent design as ../sliding-tic-tac-toe/q_learning.py. The state hash
format (board + phase + current_player) is deliberately unchanged - here
`phase` is always the literal string 'placement', which is exactly what
makes a Q-table trained on the sliding game's early placement moves directly
reusable as a warm start for this game (see bootstrap_from_sliding_agent.py):
the hash strings for equivalent early-game boards come out byte-identical.

Reward scheme is the plain textbook version - reward only at the terminal
move (win=+1, loss=-1, draw=0), nothing shaped in along the way.
"""

import pickle
import random
from collections import defaultdict

from game_engine import TicTacToe, X, O

WIN_REWARD = 1
LOSS_REWARD = -1
DRAW_REWARD = 0


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.games_played = 0
        self.wins = {X: 0, O: 0}
        self.draws = 0

    def hash_state(self, game):
        """State = board + phase + whose turn it is - identical format to
        the sliding game's agent, so early-game states hash identically."""
        return f"{tuple(game.board)}_{game.phase}_{game.current_player}"

    def get_q_value(self, state_hash, move):
        return self.q_table[state_hash][move]

    def choose_action(self, game, greedy=False):
        """Epsilon-greedy move choice. Pass greedy=True to always exploit
        (e.g. when playing for real, or evaluating trained strength)."""
        valid_moves = game.get_valid_moves()

        if not greedy and random.random() < self.exploration_rate:
            return random.choice(valid_moves)

        state_hash = self.hash_state(game)
        best_move = valid_moves[0]
        best_q = self.get_q_value(state_hash, best_move)
        for move in valid_moves[1:]:
            q = self.get_q_value(state_hash, move)
            if q > best_q:
                best_q = q
                best_move = move
        return best_move

    def train_from_self_play(self, num_games=5000, epsilon_start=0.3, epsilon_end=0.05):
        """Both players driven by the same Q-table (true self-play), with
        exploration decaying linearly from epsilon_start to epsilon_end."""
        print(f"Training Q-learning agent with {num_games} self-play games...")

        for game_num in range(num_games):
            progress = game_num / max(1, num_games - 1)
            self.exploration_rate = epsilon_start + (epsilon_end - epsilon_start) * progress

            game = TicTacToe()
            history = {X: [], O: []}

            while not game.game_over:
                player = game.current_player
                state_hash = self.hash_state(game)
                move = self.choose_action(game)
                history[player].append((state_hash, move))
                game.make_move(move)

            for player in (X, O):
                self.update_q_values(history[player], game.winner, player)

            self.games_played += 1
            if game.winner == -1:
                self.draws += 1
            else:
                self.wins[game.winner] += 1

            if (game_num + 1) % 500 == 0:
                print(f"Completed {game_num + 1} games (exploration_rate={self.exploration_rate:.3f}, "
                      f"Q-table size={len(self.q_table)})")

    def update_q_values(self, history, winner, player):
        """Backward pass: G starts at the terminal reward and gets
        discounted one step at a time as we walk back through this
        player's moves - the only reward signal is the final outcome."""
        if not history:
            return

        if winner == player:
            terminal_reward = WIN_REWARD
        elif winner == -1:
            terminal_reward = DRAW_REWARD
        else:
            terminal_reward = LOSS_REWARD

        future_return = terminal_reward
        for state_hash, move in reversed(history):
            current_q = self.get_q_value(state_hash, move)
            new_q = current_q + self.learning_rate * (future_return - current_q)
            self.q_table[state_hash][move] = new_q
            future_return = self.discount_factor * future_return

    def save_model(self, filename):
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'games_played': self.games_played,
            'wins': self.wins,
            'draws': self.draws,
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.exploration_rate = model_data['exploration_rate']
        self.games_played = model_data['games_played']
        self.wins = model_data['wins']
        self.draws = model_data['draws']
        print(f"Model loaded from {filename}")

    def get_training_stats(self):
        total = self.games_played
        return {
            'total_games': total,
            'wins_X': self.wins[X],
            'wins_O': self.wins[O],
            'draws': self.draws,
            'q_table_size': len(self.q_table),
        }


def evaluate_vs_random(agent, num_games=200):
    """Play the trained agent (as X, fully greedy - no exploration) against
    a random opponent (as O)."""
    results = {'agent': 0, 'random': 0, 'draw': 0}

    for _ in range(num_games):
        game = TicTacToe()
        while not game.game_over:
            if game.current_player == X:
                move = agent.choose_action(game, greedy=True)
            else:
                move = random.choice(game.get_valid_moves())
            game.make_move(move)

        if game.winner == X:
            results['agent'] += 1
        elif game.winner == O:
            results['random'] += 1
        else:
            results['draw'] += 1

    return results


def demonstrate_q_learning():
    print("=== CLASSIC TIC-TAC-TOE Q-LEARNING ===")
    print()

    agent = QLearningAgent()

    print("Training from self-play...")
    agent.train_from_self_play(num_games=5000)
    print()

    stats = agent.get_training_stats()
    print("Training stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("Evaluating trained agent (greedy) vs a random opponent, 200 games...")
    results = evaluate_vs_random(agent, num_games=200)
    print(f"  Results: {results}")
    print(f"  Win rate vs random: {results['agent'] / 200 * 100:.1f}%")
    print()

    agent.save_model('q_learning_model.pkl')


if __name__ == "__main__":
    demonstrate_q_learning()
