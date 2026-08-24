"""
Oware Learning System
Implementation of learning mechanisms for Oware AI
"""

import random
import pickle
from collections import defaultdict
from oware_engine import OwareGame

class OwareLearningAI:
    def __init__(self, depth=3):
        self.depth = depth
        self.player_id = 0
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q(s,a) values
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        
        # Store game history for learning
        self.game_history = []
        
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
    
    def hash_state(self, game):
        """Create a hashable representation of game state"""
        # Convert board to tuple for hashing
        board_tuple = tuple(game.board)
        score_tuple = tuple(game.scores)
        return f"{board_tuple}_{score_tuple}_{game.current_player}"
    
    def train_from_games(self, num_games=100):
        """Train AI by playing many games against itself"""
        print(f"Training AI with {num_games} self-play games...")
        
        for game_num in range(num_games):
            game = OwareGame()
            game_history = []
            
            # Play a complete game
            while not game.game_over:
                # Get move (using Q-learning)
                if game.current_player == self.player_id:
                    move = self.get_best_move_qlearning(game)
                else:
                    # Opponent plays randomly for training
                    valid_moves = game.get_valid_moves()
                    move = random.choice(valid_moves) if valid_moves else None
                
                if move is None:
                    break
                    
                # Record state and action
                state_hash = self.hash_state(game)
                game_history.append((state_hash, move))
                
                # Make move
                try:
                    game.make_move(move)
                except ValueError:
                    break  # Invalid move
            
            # Update Q-values based on game outcome
            self.update_q_values(game_history, game)
            
            if (game_num + 1) % 20 == 0:
                print(f"Completed {game_num + 1} training games")
    
    def update_q_values(self, game_history, final_game):
        """Update Q-values based on game outcome"""
        # Get reward based on game result
        reward = 0
        if final_game.winner == self.player_id:
            reward = 100  # Win
        elif final_game.winner == 1 - self.player_id:
            reward = -100  # Loss
        else:
            reward = 10  # Draw or ongoing game
        
        # Update Q-values from end to beginning
        for i in range(len(game_history) - 1, -1, -1):
            state_hash, action = game_history[i]
            
            # Q-learning update formula:
            # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            current_q = self.get_q_value(state_hash, action)
            new_q = current_q + self.learning_rate * (reward - current_q)
            self.set_q_value(state_hash, action, new_q)
            
            # Reduce reward for previous states
            reward *= self.discount_factor
    
    def save_model(self, filename):
        """Save trained model"""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate
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
            
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}")

def demonstrate_learning():
    """Demonstrate the learning process"""
    print("=== OWARE LEARNING DEMONSTRATION ===")
    print()
    
    # Create learning AI
    ai = OwareLearningAI(depth=2)
    
    print("Initial Q-table size:", len(ai.q_table))
    print("Exploration rate:", ai.exploration_rate)
    print("Learning rate:", ai.learning_rate)
    print()
    
    # Train the AI
    print("Starting training process...")
    ai.train_from_games(50)
    
    print("Training complete!")
    print("Final Q-table size:", len(ai.q_table))
    print()
    
    # Show some sample Q-values
    game = OwareGame()
    state_hash = ai.hash_state(game)
    valid_moves = game.get_valid_moves()
    
    print("Sample Q-values for initial state:")
    for move in valid_moves[:3]:  # Show first 3 moves
        q_value = ai.get_q_value(state_hash, move)
        print(f"  Move {move}: Q-value = {q_value:.2f}")
    
    print()
    print("Learning process complete!")
    print("The AI learns by:")
    print("1. Playing many games against itself")
    print("2. Recording state-action pairs")
    print("3. Updating Q-values based on outcomes")
    print("4. Improving decisions over time through experience")

if __name__ == "__main__":
    demonstrate_learning()