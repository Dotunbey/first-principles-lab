"""
Oware Reward System Design
Detailed explanation of how rewards are calculated
"""

def explain_reward_system():
    print("=== OWARE REWARD SYSTEM DESIGN ===")
    print()
    
    print("REWARD STRUCTURE FOR Q-LEARNING")
    print("-" * 40)
    print()
    
    print("1. WINNING REWARDS:")
    print("   +100 for winning a game")
    print("   +50 for capturing seeds")
    print("   +10 for controlling center houses")
    print()
    
    print("2. LOSING PENALTIES:")  
    print("   -100 for losing a game")
    print("   -50 for losing captured seeds")
    print("   -10 for poor house control")
    print()
    
    print("3. MEDIUM TERM REWARDS:")
    print("   +5 for each seed in own scoring area")
    print("   -5 for each seed in opponent scoring area")
    print("   +2 for each move that creates a capture opportunity")
    print()
    
    print("4. EXAMPLE SCENARIO:")
    print("   Game state: [4,4,4,4,4,4,4,4,4,4,4,4]")
    print("   Player 0 makes move 0")
    print("   Result: Player 0 captures 4 seeds from house 4")
    print("   Reward breakdown:")
    print("     +50 for capture")
    print("     +10 for center control")
    print("     +100 for game win")
    print("     Total: +160")
    print()
    
    print("5. REWARD FORMULA:")
    print("   R = W * Win_Penalty + C * Capture_Value + S * Score_Value")
    print("   Where:")
    print("     W = 1 if win, -1 if loss")
    print("     C = number of seeds captured")
    print("     S = score difference")
    print()
    
    print("6. IMPLEMENTATION IN CODE:")
    print("   # In update_q_values method:")
    print("   reward = 0")
    print("   if final_game.winner == self.player_id:")
    print("       reward = 100  # Win")
    print("   elif final_game.winner == 1 - self.player_id:")
    print("       reward = -100  # Loss")
    print("   else:")
    print("       reward = 10  # Draw")
    print()
    
    print("7. ADVANCED REWARDS:")
    print("   - Timing rewards (early wins better)")
    print("   - Strategy rewards (control vs aggression)")
    print("   - Risk/reward balancing")

def show_reward_calculation():
    print("\n=== REWARD CALCULATION EXAMPLE ===")
    print()
    
    print("Example game scenario:")
    print("Initial state: [4,4,4,4,4,4,4,4,4,4,4,4]")
    print("Player 0 moves from house 0")
    print("Distribution: 0->1, 1->2, 2->3, 3->4")
    print("House 4 has 4 seeds, last seed lands in house 4")
    print("House 4 originally had 4 seeds -> now has 5")
    print("No capture occurs (only 2-3 seeds trigger capture)")
    print()
    
    print("Reward calculation:")
    print("  Game result: Draw")
    print("  Reward: 10 (neutral draw)")
    print()
    
    print("Another scenario:")
    print("Player 1 moves from house 7, last seed lands in house 8")
    print("House 8 has 4 seeds, captures 4 seeds")
    print("  Capture reward: +50") 
    print("  Game result: Player 1 wins")
    print("  Game reward: +100")
    print("  Total reward: +150")

def show_reward_design_principles():
    print("\n=== REWARD DESIGN PRINCIPLES ===")
    print()
    
    print("1. CLEAR OBJECTIVES:")
    print("   - Win the game (+100)")
    print("   - Capture seeds (+50 per capture)")
    print("   - Control key positions (+10 per center house)")
    print()
    
    print("2. BALANCED PENALTIES:")
    print("   - Loss penalty (-100)")
    print("   - Poor moves (-10)")
    print("   - Risky plays (-5)")
    print()
    
    print("3. GRADIENT REWARDS:")
    print("   - Small improvements get small rewards")
    print("   - Major strategic gains get large rewards")
    print("   - Encourage progressive improvement")
    print()
    
    print("4. CONSISTENT MEASUREMENT:")
    print("   - Same reward for same strategic outcome")
    print("   - Clear reward hierarchy")
    print("   - Predictable learning signals")

if __name__ == "__main__":
    explain_reward_system()
    show_reward_calculation()
    show_reward_design_principles()