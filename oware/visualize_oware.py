"""
Oware Game Visualization
Visualizing the actual game play with clear board states
"""

from oware_engine import OwareGame
import time

class OwareVisualizer:
    def __init__(self):
        self.game = None
    
    def display_board_visual(self, game_state):
        """Display board in a more readable format"""
        print("="*50)
        print("OWARE GAME BOARD")
        print("="*50)
        
        # Display upper row (player 1's side)
        print("Player 1's side:")
        print("  ", end="")
        for i in range(5, -1, -1):  # 5 to 0
            print(f"{game_state.board[i]:2d} ", end="")
        print()
        
        print("  ", end="")
        for i in range(5, -1, -1):  # 5 to 0
            print(f" {i+1} ", end="")
        print()
        
        # Display scores
        print(f"Score: Player 0: {game_state.scores[0]} | Player 1: {game_state.scores[1]}")
        
        # Display lower row (player 2's side)
        print("  ", end="")
        for i in range(6, 12):  # 6 to 11
            print(f" {i-5} ", end="")
        print()
        
        print("  ", end="")
        for i in range(6, 12):  # 6 to 11
            print(f"{game_state.board[i]:2d} ", end="")
        print()
        
        print("Player 2's side:")
        print("="*50)
    
    def play_demo_game(self):
        """Play a demo game showing each step clearly"""
        print("Starting Oware Demo Game")
        print("Player 0 (top) vs Player 1 (bottom)")
        print()
        
        self.game = OwareGame()
        
        # Play first few moves manually to show clear progression
        moves = [
            (0, "Player 0 moves from house 0"),
            (7, "Player 1 moves from house 7 (which is house 1 on their side)"),
            (1, "Player 0 moves from house 1"),
            (8, "Player 1 moves from house 8"),
            (2, "Player 0 moves from house 2")
        ]
        
        for move_idx, (house, description) in enumerate(moves):
            print(f"\n--- MOVE {move_idx + 1} ---")
            print(description)
            print()
            
            self.display_board_visual(self.game)
            time.sleep(1)  # Pause to visualize
            
            try:
                self.game.make_move(house)
                print(f"Move completed! Last house: {house}")
                print(f"Current player: Player {self.game.current_player}")
            except ValueError as e:
                print(f"Error: {e}")
                break
                
            time.sleep(1)
        
        print("\n--- FINAL STATE ---")
        self.display_board_visual(self.game)
        
        if self.game.game_over:
            if self.game.winner == -1:
                print("Game ended in a draw!")
            else:
                print(f"Player {self.game.winner} wins!")
        else:
            print("Game continues...")

def main():
    visualizer = OwareVisualizer()
    visualizer.play_demo_game()

if __name__ == "__main__":
    main()