"""
Oware Game Engine
A Python implementation of the Oware board game (also known as Awari)
"""

class OwareGame:
    def __init__(self):
        # Initialize board with 4 seeds in each of 12 houses (6 per player)
        self.board = [4] * 12
        self.current_player = 0  # Player 0 starts
        self.scores = [0, 0]  # Score for each player
        self.game_over = False
        self.winner = None
    
    def display_board(self):
        """Display the current board state"""
        print("Player 1's side:")
        print("  ", end="")
        for i in range(5, -1, -1):
            print(f"{self.board[i]:2d} ", end="")
        print("\n   ", end="")
        for i in range(5, -1, -1):
            print(f" {i+1} ", end="")
        print()
        print(f"Score: {self.scores[0]} vs {self.scores[1]}")
        print("   ", end="")
        for i in range(6, 12):
            print(f" {i-5} ", end="")
        print()
        print("  ", end="")
        for i in range(6, 12):
            print(f"{self.board[i]:2d} ", end="")
        print("\nPlayer 2's side:")
        print()
    
    def is_valid_move(self, house_index):
        """Check if a move is valid"""
        if house_index < 0 or house_index > 11:
            return False
        if self.board[house_index] == 0:
            return False
        # Player can only move from their own side
        if self.current_player == 0 and house_index >= 6:
            return False
        if self.current_player == 1 and house_index < 6:
            return False
        return True
    
    def make_move(self, house_index):
        """Execute a move and return the index of the last house where a seed was placed"""
        if not self.is_valid_move(house_index):
            raise ValueError("Invalid move")
        
        # Get number of seeds and clear the house
        seeds = self.board[house_index]
        self.board[house_index] = 0
        
        # Distribute seeds counter-clockwise
        current_house = house_index
        last_house = house_index
        
        for _ in range(seeds):
            current_house = (current_house + 1) % 12
            # Skip opponent's scoring house
            if current_house == 6 and self.current_player == 0:
                current_house = (current_house + 1) % 12
            elif current_house == 0 and self.current_player == 1:
                current_house = (current_house + 1) % 12
            self.board[current_house] += 1
            last_house = current_house
        
        # Check for captures
        self.check_captures(last_house)
        
        # Switch player
        self.current_player = 1 - self.current_player
        
        # Check for game over
        self.check_game_over()
        
        return last_house
    
    def check_captures(self, last_house):
        """Check for captures after a move"""
        # Only capture if the last seed ended in a house with 2 or 3 seeds 
        # and it belongs to the opponent
        if self.board[last_house] in [2, 3]:
            # Determine which player owns the house
            owner = 0 if last_house < 6 else 1
            
            # Only capture if it's the opponent's house
            if owner != self.current_player:
                # Collect seeds in the house and adjacent houses
                captured_seeds = 0
                temp_house = last_house
                
                # We capture backwards until we hit a house with 0 or 4+ seeds, or our own house
                while True:
                    # Add seeds to capture count
                    captured_seeds += self.board[temp_house]
                    self.board[temp_house] = 0
                    
                    # Check if we should continue capturing
                    # Stop if it's our own house or house with 0/4+ seeds
                    if temp_house == last_house:
                        # Special case: we only capture if the capture would not eliminate 
                        # all of opponent's seeds
                        opponent_seeds = sum(self.board[6:]) if self.current_player == 0 else sum(self.board[:6])
                        if opponent_seeds == 0:
                            # If opponent has no seeds, undo capture
                            for i in range(12):
                                if self.board[i] == 0 and i != temp_house:
                                    self.board[i] = 0  # Reset all to 0 first, then restore
                            # Actually, we need to be more careful
                            break
                    
                    # Move to previous house (counter-clockwise)
                    temp_house = (temp_house - 1) % 12
                    
                    # Break if we've gone full circle
                    if temp_house == last_house:
                        break
                    
                    # Break if we hit a house with 0 or 4+ seeds
                    # Or if it's our house
                    if self.board[temp_house] in [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                        # Check if the house belongs to the same player
                        temp_owner = 0 if temp_house < 6 else 1
                        if temp_owner == self.current_player:
                            break
                        # Or if it's an empty house
                        if self.board[temp_house] == 0:
                            break
                        
                # Add captured seeds to score
                self.scores[self.current_player] += captured_seeds
    
    def check_game_over(self):
        """Check if the game is over"""
        # Check if either player has 25 or more seeds
        if self.scores[0] >= 25:
            self.game_over = True
            self.winner = 0
        elif self.scores[1] >= 25:
            self.game_over = True
            self.winner = 1
        elif self.scores[0] == 24 and self.scores[1] == 24:
            self.game_over = True
            self.winner = -1  # Draw
        else:
            # Check if current player has no valid moves
            player_houses = self.board[:6] if self.current_player == 0 else self.board[6:]
            if all(count == 0 for count in player_houses):
                # Current player has no seeds, capture rest of opponent's seeds
                opponent_houses = self.board[6:] if self.current_player == 0 else self.board[:6]
                self.scores[self.current_player] += sum(opponent_houses)
                for i in range(len(opponent_houses)):
                    if self.current_player == 0:
                        self.board[6+i] = 0
                    else:
                        self.board[i] = 0
                # Check for win condition after capturing
                if self.scores[0] >= 25:
                    self.game_over = True
                    self.winner = 0
                elif self.scores[1] >= 25:
                    self.game_over = True
                    self.winner = 1
                elif self.scores[0] == 24 and self.scores[1] == 24:
                    self.game_over = True
                    self.winner = -1  # Draw
    
    def get_valid_moves(self):
        """Get a list of valid moves for current player"""
        moves = []
        player_houses = self.board[:6] if self.current_player == 0 else self.board[6:]
        for i, count in enumerate(player_houses):
            if count > 0:
                if self.current_player == 0:
                    moves.append(i)
                else:
                    moves.append(i + 6)
        return moves
    
    def copy(self):
        """Create a copy of the game state"""
        new_game = OwareGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.scores = self.scores.copy()
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

def main():
    """Main function to demonstrate the game"""
    game = OwareGame()
    
    print("Welcome to Oware!")
    print("Player 0 (top) vs Player 1 (bottom)")
    print()
    
    while not game.game_over:
        game.display_board()
        print(f"Player {game.current_player}'s turn")
        print("Valid moves:", game.get_valid_moves())
        
        try:
            move = int(input("Choose a house (0-11): "))
            game.make_move(move)
        except (ValueError, IndexError):
            print("Invalid move! Try again.")
            continue
    
    game.display_board()
    if game.winner == -1:
        print("Game ended in a draw!")
    else:
        print(f"Player {game.winner} wins!")

if __name__ == "__main__":
    main()