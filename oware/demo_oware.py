"""
Oware Game Demo
Demonstrating the Oware game engine and AI player
"""

from oware_engine import OwareGame
from oware_ai import OwareAI

def demo_game():
    """Demonstrate a simple game between two AIs"""
    print("=== Oware Game Demo ===")
    print("Player 0 (Human) vs Player 1 (AI)")
    print()
    
    # Create game
    game = OwareGame()
    ai = OwareAI(depth=2)
    
    # Set the AI to play as player 1
    ai.player_id = 1
    
    round_count = 0
    max_rounds = 20  # Prevent infinite games
    
    while not game.game_over and round_count < max_rounds:
        print(f"--- Round {round_count + 1} ---")
        game.display_board()
        
        if game.current_player == 0:
            # Human player (we'll simulate a move)
            print("Player 0's turn - Simulating move from house 0")
            game.make_move(0)
        else:
            # AI player
            print("Player 1's turn - AI thinking...")
            best_move = ai.get_best_move(game)
            print(f"AI chooses house {best_move}")
            game.make_move(best_move)
            
        round_count += 1
        print()
    
    # Show final result
    game.display_board()
    if game.game_over:
        if game.winner == -1:
            print("Game ended in a draw!")
        else:
            print(f"Player {game.winner} wins!")
    else:
        print("Game reached maximum rounds")

if __name__ == "__main__":
    demo_game()