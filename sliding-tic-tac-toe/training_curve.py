"""
Visualize the Q-learning agent's strength over the course of training.

Trains in chunks, pausing every `checkpoint_interval` games to measure the
agent's (greedy, no exploration) win rate against a random opponent, then
plots that as a curve - this is the "how long did it take to catch up"
picture, with MCTS's instant 100% drawn in as a flat reference line.
"""

import matplotlib.pyplot as plt

from game_engine import SlidingTicTacToe, X, O
from q_learning import QLearningAgent, evaluate_vs_random


def train_with_checkpoints(total_games=6000, checkpoint_interval=250,
                            epsilon_start=0.3, epsilon_end=0.05, eval_games=100):
    agent = QLearningAgent()
    checkpoints = []
    win_rates = []

    for game_num in range(total_games):
        progress = game_num / max(1, total_games - 1)
        agent.exploration_rate = epsilon_start + (epsilon_end - epsilon_start) * progress

        game = SlidingTicTacToe()
        history = {X: [], O: []}
        while not game.game_over:
            player = game.current_player
            state_hash = agent.hash_state(game)
            move = agent.choose_action(game)
            history[player].append((state_hash, move))
            game.make_move(move)

        for player in (X, O):
            agent.update_q_values(history[player], game.winner, player)

        agent.games_played += 1
        if game.winner == -1:
            agent.draws += 1
        else:
            agent.wins[game.winner] += 1

        if (game_num + 1) % checkpoint_interval == 0:
            results = evaluate_vs_random(agent, num_games=eval_games)
            win_rate = results['agent'] / eval_games * 100
            checkpoints.append(game_num + 1)
            win_rates.append(win_rate)
            print(f"After {game_num + 1:>5} games: win rate vs random = {win_rate:5.1f}%  "
                  f"(Q-table size={len(agent.q_table)})")

    return agent, checkpoints, win_rates


def plot_learning_curve(checkpoints, win_rates, save_path='learning_curve.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, win_rates, marker='o', label='Q-learning agent')
    plt.axhline(100, color='gray', linestyle='--', linewidth=1,
                label='MCTS (100% - needs zero training games)')
    plt.xlabel('Self-play games trained')
    plt.ylabel('Win rate vs random opponent (%)')
    plt.title('Q-learning agent strength over training (Sliding Tic-Tac-Toe)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nSaved chart to {save_path}")
    plt.show()


if __name__ == "__main__":
    agent, checkpoints, win_rates = train_with_checkpoints()
    agent.save_model('q_learning_model.pkl')
    plot_learning_curve(checkpoints, win_rates)
