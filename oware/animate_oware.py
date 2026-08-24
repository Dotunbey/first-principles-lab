"""
Oware Game Animation
Show the actual seed movement in each move
"""

from oware_engine import OwareGame

class OwareAnimator:
    def __init__(self):
        self.game = None
    
    def animate_move(self, house_index):
        """Animate a move showing seed distribution"""
        print(f"\n--- ANIMATING MOVE FROM HOUSE {house_index} ---")
        print(f"Player {self.game.current_player} moves seeds from house {house_index}")
        print()
        
        # Show initial board state
        self.show_board_state("BEFORE MOVE")
        
        # Get the number of seeds and clear the house
        seeds = self.game.board[house_index]
        print(f"House {house_index} contains {seeds} seeds")
        print(f"Clearing house {house_index}...")
        
        # Show the move process
        current_house = house_index
        last_house = house_index
        
        print("\nDistributing seeds counter-clockwise:")
        
        for i in range(seeds):
            current_house = (current_house + 1) % 12
            # Skip opponent's scoring house
            if current_house == 6 and self.game.current_player == 0:
                current_house = (current_house + 1) % 12
                print(f"  Skipping opponent's scoring house (house 6)")
            elif current_house == 0 and self.game.current_player == 1:
                current_house = (current_house + 1) % 12
                print(f"  Skipping opponent's scoring house (house 0)")
            
            self.game.board[current_house] += 1
            last_house = current_house
            print(f"  Seed {i+1}: Placed in house {current_house}")
        
        print(f"\nLast house where seed was placed: {last_house}")
        
        # Show final state
        self.show_board_state("AFTER MOVE")
        
        # Check for captures
        if self.game.board[last_house] in [2, 3]:
            print(f"Capture check: House {last_house} has {self.game.board[last_house]} seeds")
            print("Checking for capture opportunity...")
            # In a real implementation, we would perform the capture logic here
        else:
            print(f"No capture: House {last_house} has {self.game.board[last_house]} seeds")
        
        # Switch player
        self.game.current_player = 1 - self.game.current_player
        print(f"Switching to Player {self.game.current_player}")
    
    def show_board_state(self, title):
        """Display current board state"""
        print(f"\n{title}:")
        print("Player 1's side:")
        print("  ", end="")
        for i in range(5, -1, -1):  # 5 to 0
            print(f"{self.game.board[i]:2d} ", end="")
        print()
        
        print("  ", end="")
        for i in range(5, -1, -1):  # 5 to 0
            print(f" {i+1} ", end="")
        print()
        
        print(f"Score: Player 0: {self.game.scores[0]} | Player 1: {self.game.scores[1]}")
        
        print("  ", end="")
        for i in range(6, 12):  # 6 to 11
            print(f" {i-5} ", end="")
        print()
        
        print("  ", end="")
        for i in range(6, 12):  # 6 to 11
            print(f"{self.game.board[i]:2d} ", end="")
        print()
        
        print("Player 2's side:")
        print("-"*30)
    
    def run_animation_demo(self):
        """Run a demo showing seed distribution"""
        print("OWARE SEED DISTRIBUTION ANIMATION")
        print("="*40)
        
        self.game = OwareGame()
        
        # Show initial state
        self.show_board_state("INITIAL STATE")
        
        # Animate several moves
        moves = [(0, "Player 0 moves from house 0"), 
                 (7, "Player 1 moves from house 7")]
        
        for house, description in moves:
            print(f"\n{description}")
            self.animate_move(house)
            input("\nPress Enter to continue...")

def main():
    animator = OwareAnimator()
    animator.run_animation_demo()

if __name__ == "__main__":
    main()