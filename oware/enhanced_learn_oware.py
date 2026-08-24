"""
Enhanced Oware Learning System
With multiple learning approaches and expert game integration
"""

import random
import pickle
import json
from collections import defaultdict
from oware_engine import OwareGame

CENTER_HOUSES = {2, 3, 8, 9}  # middle houses of each side, per reward_system.py design
CAPTURE_BONUS = 50
CENTER_BONUS = 10


class EnhancedOwareLearningAI:
    def __init__(self, depth=3):
        self.depth = depth
        self.player_id = 0
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.policy_network = {}  # Simple policy network
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

        # Learning statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
    def hash_state(self, game):
        """Create a hashable representation of game state"""
        board_tuple = tuple(game.board)
        score_tuple = tuple(game.scores)
        return f"{board_tuple}_{score_tuple}_{game.current_player}"
    
    def get_q_value(self, state_hash, action):
        """Get Q-value for state-action pair"""
        return self.q_table[state_hash][action]
    
    def set_q_value(self, state_hash, action, value):
        """Set Q-value for state-action pair"""
        self.q_table[state_hash][action] = value
    
    def get_best_move_qlearning(self, game):
        """Get best move using Q-learning with exploration"""
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
            
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: choose random move
            return random.choice(valid_moves)
        else:
            # Exploit: choose best move based on Q-values
            state_hash = self.hash_state(game)
            best_move = valid_moves[0]
            best_q_value = self.get_q_value(state_hash, best_move)
            
            for move in valid_moves[1:]:
                q_value = self.get_q_value(state_hash, move)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_move = move
                    
            return best_move
    
    def compute_step_reward(self, player, seeds_captured, last_house):
        """Shaped intermediate reward for one move, per reward_system.py design"""
        reward = 0
        if seeds_captured > 0:
            reward += CAPTURE_BONUS
        if last_house in CENTER_HOUSES:
            reward += CENTER_BONUS
        return reward

    def train_from_self_play(self, num_games=100, epsilon_start=0.3, epsilon_end=0.05):
        """Train AI by playing both sides against itself (true self-play),
        with exploration decaying from epsilon_start to epsilon_end."""
        print(f"Training AI with {num_games} self-play games...")

        for game_num in range(num_games):
            # Decay exploration over the course of training
            progress = game_num / max(1, num_games - 1)
            self.exploration_rate = epsilon_start + (epsilon_end - epsilon_start) * progress

            game = OwareGame()
            history = {0: [], 1: []}

            # Play a complete game, both players driven by the same Q-table
            while not game.game_over:
                player = game.current_player
                move = self.get_best_move_qlearning(game)

                if move is None:
                    break

                state_hash = self.hash_state(game)
                scores_before = game.scores.copy()

                try:
                    last_house = game.make_move(move)
                except ValueError:
                    break  # Invalid move

                seeds_captured = game.scores[player] - scores_before[player]
                step_reward = self.compute_step_reward(player, seeds_captured, last_house)
                history[player].append((state_hash, move, step_reward))

            # Update Q-values for both players based on their own outcome
            for player in (0, 1):
                self.update_q_values(history[player], game, player)

            # Update statistics (tracked from self.player_id's perspective)
            self.games_played += 1
            if game.winner == self.player_id:
                self.wins += 1
            elif game.winner == 1 - self.player_id:
                self.losses += 1
            else:
                self.draws += 1

            if (game_num + 1) % 20 == 0:
                print(f"Completed {game_num + 1} training games (exploration_rate={self.exploration_rate:.3f})")

    def train_from_expert_games(self, expert_games_data):
        """Train AI by replaying real expert game records move-by-move,
        crediting each state-action pair with the game's actual outcome
        plus the same shaped rewards used in self-play."""
        print(f"Training from {len(expert_games_data)} expert games...")

        for game_num, game_data in enumerate(expert_games_data):
            game = OwareGame()
            if 'board_start' in game_data:
                game.board = game_data['board_start'][:12]

            history = {0: [], 1: []}

            for move in game_data.get('moves', []):
                if game.game_over or not game.is_valid_move(move):
                    break

                player = game.current_player
                state_hash = self.hash_state(game)
                scores_before = game.scores.copy()

                last_house = game.make_move(move)

                seeds_captured = game.scores[player] - scores_before[player]
                step_reward = self.compute_step_reward(player, seeds_captured, last_house)
                history[player].append((state_hash, move, step_reward))

            for player in (0, 1):
                self.update_q_values(history[player], game, player)

            print(f"Processed expert game {game_num + 1} ({game_data.get('id', '?')})")

    def update_q_values(self, game_history, final_game, player):
        """Update Q-values for one player's moves using their shaped
        per-step rewards plus the game's terminal outcome, propagated
        backward through the discounted return."""
        if not game_history:
            return

        if final_game.winner == player:
            terminal_reward = 100  # Win
        elif final_game.winner == -1:
            terminal_reward = 10  # Draw
        else:
            terminal_reward = -100  # Loss

        # Backward pass computing the discounted return G at each step:
        # G_t = r_t + gamma * G_{t+1}, seeded with the terminal outcome
        future_return = terminal_reward
        for state_hash, action, step_reward in reversed(game_history):
            future_return = step_reward + self.discount_factor * future_return

            # Q-learning update formula:
            # Q(s,a) = Q(s,a) + α[G - Q(s,a)]
            current_q = self.get_q_value(state_hash, action)
            new_q = current_q + self.learning_rate * (future_return - current_q)
            self.set_q_value(state_hash, action, new_q)
    
    def save_model(self, filename):
        """Save trained model"""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.exploration_rate = model_data['exploration_rate']
            self.games_played = model_data['games_played']
            self.wins = model_data['wins']
            self.losses = model_data['losses']
            self.draws = model_data['draws']
            
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}")
    
    def get_training_stats(self):
        """Get current training statistics"""
        total_games = self.games_played
        win_rate = (self.wins / total_games * 100) if total_games > 0 else 0
        loss_rate = (self.losses / total_games * 100) if total_games > 0 else 0
        draw_rate = (self.draws / total_games * 100) if total_games > 0 else 0
        
        return {
            'total_games': total_games,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate
        }

def demonstrate_enhanced_learning():
    """Demonstrate enhanced learning capabilities"""
    print("=== ENHANCED OWARE LEARNING SYSTEM ===")
    print()
    
    # Create enhanced learning AI
    ai = EnhancedOwareLearningAI(depth=2)
    
    print("Initial training statistics:")
    stats = ai.get_training_stats()
    print(f"  Games played: {stats['total_games']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print()
    
    # Train with self-play
    print("1. Training with self-play...")
    ai.train_from_self_play(30)
    
    print("\nTraining statistics after self-play:")
    stats = ai.get_training_stats()
    print(f"  Games played: {stats['total_games']}")
    print(f"  Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print()
    
    # Load real expert games recorded in oware_games_database.json
    print("2. Training from recorded expert games...")
    with open('oware_games_database.json') as f:
        expert_games = json.load(f)

    ai.train_from_expert_games(expert_games)
    print("Expert game training complete!")
    print()
    
    # Save the model
    ai.save_model('oware_model.pkl')
    
    print("3. Model saved successfully!")
    print("\nLearning Process Summary:")
    print("• Self-play training: AI improves by playing against itself")
    print("• Q-learning: Uses rewards to update strategy values")
    print("• Expert game integration: Learns from master strategies")
    print("• Continuous improvement: Statistics track progress over time")
    print("• Model persistence: Trained models can be saved and loaded")

if __name__ == "__main__":
    demonstrate_enhanced_learning()