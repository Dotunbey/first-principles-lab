"""
Oware Learning Process Overview
Simple version without unicode characters
"""

from enhanced_learn_oware import EnhancedOwareLearningAI
from oware_engine import OwareGame
import json
import random

def demonstrate_learning_setup():
    """Show how to set up and start the learning process"""
    print("=== OWARE LEARNING PROCESS SETUP ===")
    print()
    
    print("HOW THE LEARNING PROCESS STARTS:")
    print("-" * 40)
    print("1. Initialize the AI with learning parameters")
    print("2. Start training with self-play games")  
    print("3. Integrate expert game databases")
    print("4. Automatically improve through experience")
    print()
    
    print("SETUP STEPS:")
    print("-" * 40)
    print("Step 1: Create AI instance")
    print("  ai = EnhancedOwareLearningAI(depth=3)")
    print()
    
    print("Step 2: Start training with self-play")
    print("  ai.train_from_self_play(num_games=100)")
    print()
    
    print("Step 3: Load expert game database")
    print("  with open('oware_games_database.json') as f:")
    print("      expert_games = json.load(f)")
    print("  ai.train_from_expert_games(expert_games)")
    print()
    
    print("LEARNING METHODS YOU CAN USE:")
    print("-" * 40)
    print("Q-Learning: Core reinforcement learning method")
    print("  - Uses rewards to update strategy values")
    print("  - Updates Q-table based on game outcomes")
    print()
    
    print("Self-Play: AI improves by playing against itself")
    print("  - Generates diverse training scenarios")
    print("  - Learns through trial and error")
    print()
    
    print("Expert Game Integration: Learn from master strategies")
    print("  - Parse game databases from experts")
    print("  - Reinforce proven winning moves")
    print("  - Accelerate learning curve")
    print()

def demonstrate_simple_learning():
    """Show a simple learning example"""
    print("=== SIMPLE LEARNING DEMONSTRATION ===")
    print()
    
    # Create AI
    ai = EnhancedOwareLearningAI(depth=2)
    print("Created Oware Learning AI")
    
    # Show initial state
    print("Initial Q-table size:", len(ai.q_table))
    print("Exploration rate:", ai.exploration_rate)
    print()
    
    # Train for few games
    print("Training with 10 self-play games...")
    ai.train_from_self_play(10)
    
    # Show results
    stats = ai.get_training_stats()
    print("After training:")
    print(f"  Games played: {stats['total_games']}")
    print(f"  Wins: {stats['wins']}, Losses: {stats['losses']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print()
    
    # Save model
    ai.save_model('simple_oware_model.pkl')
    print("Model saved as simple_oware_model.pkl")
    
    print()
    print("YOU ARE NOW READY TO START LEARNING!")
    print("To continue learning:")
    print("1. Run more training games: ai.train_from_self_play(100)")
    print("2. Add expert games: ai.train_from_expert_games(your_games)")
    print("3. Load existing model: ai.load_model('simple_oware_model.pkl')")
    print("4. Test the learned AI in gameplay")

if __name__ == "__main__":
    demonstrate_learning_setup()
    demonstrate_simple_learning()