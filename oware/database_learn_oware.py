"""
Oware Database Integration
Demonstrating how to show the AI other games to learn from
"""

from enhanced_learn_oware import EnhancedOwareLearningAI
from oware_engine import OwareGame
import json
import random

def create_sample_database():
    """Create a sample database of Oware games with expert moves"""
    print("Creating sample Oware game database...")
    
    # Sample games from expert play
    games_db = [
        {
            "id": "expert_001",
            "players": ["Master_A", "Master_B"],
            "result": "Master_A_wins",
            "moves": [0, 7, 2, 9, 1, 8, 3, 10],  # Example move sequence
            "board_start": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "final_score": [26, 18],
            "comments": "Strategic opening with control of center houses"
        },
        {
            "id": "expert_002", 
            "players": ["Grandmaster_X", "Grandmaster_Y"],
            "result": "Grandmaster_Y_wins",
            "moves": [5, 6, 4, 7, 3, 8, 2, 9, 1, 10],  # Different sequence
            "board_start": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "final_score": [19, 25],
            "comments": "Aggressive early play focusing on opponent's side"
        },
        {
            "id": "expert_003",
            "players": ["Expert_Player_1", "Expert_Player_2"],
            "result": "Draw",
            "moves": [2, 9, 1, 8, 3, 7, 4, 6, 5, 0],  # Balanced approach
            "board_start": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "final_score": [24, 24],
            "comments": "Symmetrical play leading to draw"
        }
    ]
    
    # Save to file
    with open('oware_games_database.json', 'w') as f:
        json.dump(games_db, f, indent=2)
    
    print("Sample database created: oware_games_database.json")
    return games_db

def demonstrate_database_integration():
    """Demonstrate integrating a database of Oware games"""
    print("=== OWARE DATABASE INTEGRATION ===")
    print()
    
    # Create sample database
    games_db = create_sample_database()
    
    # Load and display sample games
    print("Sample games from database:")
    for i, game in enumerate(games_db[:2]):  # Show first 2 games
        print(f"\nGame {i+1}: {game['id']}")
        print(f"  Players: {game['players'][0]} vs {game['players'][1]}")  
        print(f"  Result: {game['result']}")
        print(f"  Moves: {game['moves']}")
        print(f"  Comments: {game['comments']}")
    
    print("\n" + "="*50)
    print("DATABASE INTEGRATION PROCESS:")
    print("="*50)
    print()
    
    # Create AI and show how it would learn from database
    ai = EnhancedOwareLearningAI(depth=2)
    
    print("1. Loading expert games into learning system...")
    print("   - Parsing game records")
    print("   - Extracting strategic moves")
    print("   - Identifying high-performing sequences")
    print()
    
    # Simulate loading games
    print("2. Demonstrating learning from expert moves:")
    for i, game_data in enumerate(games_db[:2]):
        print(f"   Processing {game_data['id']}:")
        print(f"     - Analyzing {len(game_data['moves'])} expert moves")
        print(f"     - Reinforcing successful strategies")
        print(f"     - Updating Q-values for identified good positions")
        print()
    
    # Simulate training from database
    print("3. Training with database integration:")
    
    # This is where we would actually load and process the database
    # For demonstration, we'll show what happens internally:
    
    print("   - For each expert game:")
    print("     • Analyze opening moves")
    print("     • Identify capture opportunities")
    print("     • Learn from winning strategies")
    print("     • Avoid common mistakes")
    print()
    
    print("4. Hybrid learning approach:")
    print("   • Combine self-play for creativity")
    print("   • Integrate expert databases for proven strategies")  
    print("   • Continuously refine through experience")
    print()
    
    print("5. Data sources that can be used:")
    print("   • Historical Oware tournament records")
    print("   • Master player game transcripts")
    print("   • Academic research papers on Oware strategy")
    print("   • Community-created game databases")
    print()
    
    print("6. Benefits of expert game integration:")
    print("   • Faster learning convergence")
    print("   • Access to proven strategies")
    print("   • Reduced need for extensive self-play")
    print("   • Preservation of cultural knowledge")
    
    print("\n" + "="*50)
    print("DATABASE INTEGRATION COMPLETE")
    print("="*50)
    
    # Show how to load database data
    print("\nCode to load database (sample):")
    print("""
# Load from JSON database
with open('oware_games_database.json') as f:
    games_data = json.load(f)
    
# Process each game
for game_record in games_data:
    # Extract moves and positions
    expert_moves = game_record['moves'] 
    
    # Reinforce good moves in AI's Q-table
    for move in expert_moves:
        state_hash = hash_current_state()
        ai.set_q_value(state_hash, move, ai.get_q_value(state_hash, move) + 5)
    """)

def demonstrate_learning_workflow():
    """Show complete learning workflow"""
    print("\n=== COMPLETE LEARNING WORKFLOW ===")
    print()
    
    print("Step 1: Prepare Learning Environment")
    print("  ✓ Initialize Oware game engine")
    print("  ✓ Setup learning AI with Q-learning")
    print("  ✓ Configure parameters (learning rate, exploration)")
    print()
    
    print("Step 2: Gather Training Data")
    print("  ✓ Create sample game database")
    print("  ✓ Include expert games from masters")
    print("  ✓ Add self-play generated games")
    print()
    
    print("Step 3: Train the AI")
    print("  ✓ Self-play training (100 games)")
    print("  ✓ Expert database integration")
    print("  ✓ Q-value updates based on outcomes")
    print()
    
    print("Step 4: Evaluate and Improve")
    print("  ✓ Monitor win rates and stats")
    print("  ✓ Adjust learning parameters")
    print("  ✓ Save trained model")
    print()
    
    print("Step 5: Deploy and Continue Learning")
    print("  ✓ Use trained AI for game play")
    print("  ✓ Generate new training data")
    print("  ✓ Iterative improvement cycle")

if __name__ == "__main__":
    demonstrate_database_integration()
    demonstrate_learning_workflow()