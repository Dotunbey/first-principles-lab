"""
Simple Oware AI Player
Uses minimax with alpha-beta pruning and basic heuristics
"""

import sys
from typing import List, Tuple
from oware_engine import OwareGame

class OwareAI:
    def __init__(self, depth: int = 3):
        self.depth = depth
        self.player_id = 0  # This will be set based on who the AI plays as
    
    def evaluate_position(self, game: OwareGame) -> float:
        """
        Evaluate the current board position
        Returns a score: positive for advantage, negative for disadvantage
        """
        # Basic evaluation function
        score_difference = game.scores[0] - game.scores[1]
        
        # Count seeds in player's houses
        player_seeds = sum(game.board[:6]) if self.player_id == 0 else sum(game.board[6:])
        opponent_seeds = sum(game.board[6:]) if self.player_id == 0 else sum(game.board[:6])
        
        # Prefer positions where we have more seeds in our houses
        seed_advantage = player_seeds - opponent_seeds
        
        # Prefer positions where opponent has fewer moves available
        opponent_moves = len(game.get_valid_moves()) if self.player_id != 0 else len([i for i in range(6) if game.board[i] > 0])
        
        # Simple heuristic: prioritize captures and having more seeds
        return score_difference * 10 + seed_advantage * 2 - opponent_moves
        
    def minimax(self, game: OwareGame, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, int]:
        """
        Minimax algorithm with alpha-beta pruning
        Returns (best_score, best_move)
        """
        # Base case: reached max depth or game over
        if depth == 0 or game.game_over:
            return self.evaluate_position(game), -1
        
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            # No valid moves, game over
            return self.evaluate_position(game), -1
            
        if maximizing_player:
            max_eval = float('-inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                # Make the move
                new_game = game.copy()
                new_game.make_move(move)
                
                # Recursive call
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                # Make the move
                new_game = game.copy()
                new_game.make_move(move)
                
                # Recursive call
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return min_eval, best_move
    
    def get_best_move(self, game: OwareGame) -> int:
        """
        Get the best move for the current player using minimax
        """
        _, best_move = self.minimax(game, self.depth, float('-inf'), float('inf'), True)
        return best_move

def play_game():
    """Play a game between human and AI"""
    game = OwareGame()
    ai = OwareAI(depth=3)
    
    print("Welcome to Oware vs AI!")
    print("You are Player 0 (top), AI is Player 1 (bottom)")
    print("Enter house numbers 0-11 to make your move")
    print()
    
    while not game.game_over:
        game.display_board()
        
        if game.current_player == 0:
            # Human player's turn
            print("Your turn (Player 0)")
            print("Valid moves:", game.get_valid_moves())
            
            try:
                move = int(input("Choose a house (0-11): "))
                if move not in game.get_valid_moves():
                    print("Invalid move! Try again.")
                    continue
                game.make_move(move)
            except (ValueError, IndexError):
                print("Invalid input! Try again.")
                continue
        else:
            # AI's turn
            print("AI is thinking...")
            ai_move = ai.get_best_move(game)
            print(f"AI chooses house {ai_move}")
            game.make_move(ai_move)
            
    # Game over
    game.display_board()
    if game.winner == -1:
        print("Game ended in a draw!")
    elif game.winner == 0:
        print("You win!")
    else:
        print("AI wins!")

if __name__ == "__main__":
    play_game()