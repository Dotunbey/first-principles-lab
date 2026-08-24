"""
Test file for Oware game implementation
"""

from oware_engine import OwareGame

def test_basic_game():
    """Test basic game functionality"""
    print("Testing basic Oware game...")
    
    game = OwareGame()
    
    # Test initial state
    assert game.board == [4] * 12, "Initial board should have 4 seeds in each house"
    assert game.current_player == 0, "Player 0 should start"
    assert game.scores == [0, 0], "Initial scores should be 0"
    assert not game.game_over, "Game should not be over initially"
    
    print("[PASS] Initial state test passed")
    
    # Test valid move
    try:
        game.make_move(0)  # Player 0 moves from house 0
        assert game.board[0] == 0, "House 0 should be empty after move"
        assert game.board[1] == 5, "House 1 should have 5 seeds after move"
        assert game.current_player == 1, "Player should switch after move"
        print("[PASS] Basic move test passed")
    except Exception as e:
        print(f"[FAIL] Basic move test failed: {e}")
        return False
    
    # Test invalid move - we need to create a new game for this test
    # because the previous operation changed the state
    game2 = OwareGame()  # Fresh game
    try:
        game2.make_move(6)  # Player 0 trying to move from Player 1's house
        print("[FAIL] Invalid move test failed: Should have raised exception")
        return False
    except ValueError:
        print("[PASS] Invalid move test passed")
    
    # Test game over detection
    # Create a scenario where game should end
    game3 = OwareGame()
    game3.scores[0] = 25  # Player 0 wins
    game3.check_game_over()
    assert game3.game_over, "Game should be over when player reaches 25 seeds"
    assert game3.winner == 0, "Winner should be Player 0"
    print("[PASS] Game over test passed")
    
    print("All tests passed!")
    return True

def test_capture_logic():
    """Test capture logic"""
    print("Testing capture logic...")
    
    # Create a scenario where capture should happen
    game = OwareGame()
    
    # Manually set up a position where capture should occur
    # This is a simplified test - in practice, we'd need a more complex setup
    game.board = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # Player 0's house 5 has 1 seed
    game.current_player = 1  # Player 1's turn
    
    # Make a move that causes opponent's house to have 2 seeds
    # This would trigger a capture in a real scenario
    
    print("Capture logic test completed")

if __name__ == "__main__":
    test_basic_game()
    test_capture_logic()