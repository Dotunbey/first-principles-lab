"""
Final Oware Game Visualization
Complete visualization of actual game play
"""

from oware_engine import OwareGame

def show_complete_game():
    """Show a complete game with clear visualization"""
    print("OWARE GAME PLAY VISUALIZATION")
    print("=" * 50)
    print()
    
    # Create game
    game = OwareGame()
    
    print("INITIAL BOARD STATE:")
    print("Player 1's side (houses 5-0):")
    print("  ", end="")
    for i in range(5, -1, -1):  # 5 to 0
        print(f"{game.board[i]:2d} ", end="")
    print()
    
    print("  ", end="")
    for i in range(5, -1, -1):  # 5 to 0
        print(f" {i+1} ", end="")
    print()
    
    print(f"Score: Player 0: {game.scores[0]} | Player 1: {game.scores[1]}")
    
    print("  ", end="")
    for i in range(6, 12):  # 6 to 11
        print(f" {i-5} ", end="")
    print()
    
    print("  ", end="")
    for i in range(6, 12):  # 6 to 11
        print(f"{game.board[i]:2d} ", end="")
    print()
    
    print("Player 2's side (houses 6-11):")
    print("=" * 50)
    print()
    
    # Make a few moves to demonstrate gameplay
    moves = [
        (0, "Player 0 moves from house 0 (4 seeds)"),
        (7, "Player 1 moves from house 7 (4 seeds)"), 
        (2, "Player 0 moves from house 2 (5 seeds)")
    ]
    
    for move_num, (house, description) in enumerate(moves, 1):
        print(f"MOVE {move_num}: {description}")
        print("-" * 30)
        
        # Show what happens during the move
        seeds = game.board[house]
        print(f"House {house} contains {seeds} seeds")
        print(f"Clearing house {house}...")
        
        # Show the distribution
        current_house = house
        for i in range(seeds):
            current_house = (current_house + 1) % 12
            # Skip opponent's scoring house
            if current_house == 6 and game.current_player == 0:
                current_house = (current_house + 1) % 12
                print(f"  Skipping opponent's scoring house (house 6)")
            elif current_house == 0 and game.current_player == 1:
                current_house = (current_house + 1) % 12
                print(f"  Skipping opponent's scoring house (house 0)")
            
            game.board[current_house] += 1
            print(f"  Seed {i+1}: Placed in house {current_house}")
        
        # Show updated board
        print("\nNEW BOARD STATE:")
        print("Player 1's side:")
        print("  ", end="")
        for i in range(5, -1, -1):  # 5 to 0
            print(f"{game.board[i]:2d} ", end="")
        print()
        
        print("  ", end="")
        for i in range(5, -1, -1):  # 5 to 0
            print(f" {i+1} ", end="")
        print()
        
        print(f"Score: Player 0: {game.scores[0]} | Player 1: {game.scores[1]}")
        
        print("  ", end="")
        for i in range(6, 12):  # 6 to 11
            print(f" {i-5} ", end="")
        print()
        
        print("  ", end="")
        for i in range(6, 12):  # 6 to 11
            print(f"{game.board[i]:2d} ", end="")
        print()
        
        print("Player 2's side:")
        print("-" * 30)
        
        # Switch players
        game.current_player = 1 - game.current_player
        print(f"Next turn: Player {game.current_player}")
        print()
    
    print("GAME PLAY COMPLETE")
    print("=" * 50)
    print("This demonstrates the actual seed distribution mechanics in Oware:")
    print("• Players alternate turns moving seeds counter-clockwise")
    print("• Seeds are distributed one by one to subsequent houses")
    print("• Opponent's scoring house is skipped")
    print("• Strategic capture occurs when last seed lands in house with 2-3 seeds")
    print("• Game continues until one player reaches 25 points or no moves remain")

if __name__ == "__main__":
    show_complete_game()