"""
Train a FRESH Q-learning agent by having it play directly against MCTS
(instead of self-play). Every single game is logged - who won, how many
moves, the agent's exploration rate at the time, the move sequence - and a
rich multi-panel dashboard tracks how the gap between the two agents closes
(or doesn't) over the course of training.

MCTS never learns or changes here - it's a fixed, strong sparring partner.
Only the Q-agent's table gets updated, from its half of each game.
"""

import json
import random

import matplotlib.pyplot as plt

from game_engine import SlidingTicTacToe, X, O
from mcts import mcts_search
from q_learning import QLearningAgent


def play_one_game(agent, mcts_iterations, q_plays):
    """q_plays: X or O - which side the Q-agent controls this game.
    Returns the finished game, the Q-agent's own (state, move) history for
    the Q-update, and the full ordered move list for both sides (for logging
    and later replay)."""
    game = SlidingTicTacToe()
    q_history = []
    full_moves = []

    while not game.game_over:
        player = game.current_player
        if player == q_plays:
            state_hash = agent.hash_state(game)
            move = agent.choose_action(game)
            q_history.append((state_hash, move))
        else:
            move = mcts_search(game, iterations=mcts_iterations)
        full_moves.append(move)
        game.make_move(move)

    return game, q_history, full_moves


def evaluate_vs_mcts(agent, mcts_iterations=150, num_games=20):
    """Greedy (no exploration) Q-agent vs MCTS, alternating sides, to
    measure the current gap without the noise of exploration."""
    wins = losses = draws = 0

    for i in range(num_games):
        q_plays = X if i % 2 == 0 else O
        game = SlidingTicTacToe()
        while not game.game_over:
            player = game.current_player
            if player == q_plays:
                move = agent.choose_action(game, greedy=True)
            else:
                move = mcts_search(game, iterations=mcts_iterations)
            game.make_move(move)

        if game.winner == -1:
            draws += 1
        elif game.winner == q_plays:
            wins += 1
        else:
            losses += 1

    return {'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / num_games * 100}


def train_vs_mcts(num_games=1000, mcts_iterations=150, checkpoint_interval=100,
                   eval_games=20, epsilon_start=0.4, epsilon_end=0.05):
    agent = QLearningAgent()  # fresh - no loaded model
    game_log = []
    checkpoints = []
    win_rates = []
    draw_rates = []

    for game_num in range(num_games):
        progress = game_num / max(1, num_games - 1)
        agent.exploration_rate = epsilon_start + (epsilon_end - epsilon_start) * progress

        q_plays = X if game_num % 2 == 0 else O
        game, q_history, full_moves = play_one_game(agent, mcts_iterations, q_plays)
        agent.update_q_values(q_history, game.winner, q_plays)

        agent.games_played += 1
        if game.winner == -1:
            outcome = 'draw'
            agent.draws += 1
        elif game.winner == q_plays:
            outcome = 'q_win'
            agent.wins[q_plays] += 1
        else:
            outcome = 'q_loss'
            opponent = O if q_plays == X else X
            agent.wins[opponent] += 1

        game_log.append({
            'game_num': game_num + 1,
            'q_plays': 'X' if q_plays == X else 'O',
            'outcome': outcome,
            'move_count': len(full_moves),
            'exploration_rate': round(agent.exploration_rate, 4),
            'moves': [list(m) if isinstance(m, tuple) else m for m in full_moves],
        })

        if (game_num + 1) % checkpoint_interval == 0:
            results = evaluate_vs_mcts(agent, mcts_iterations, eval_games)
            checkpoints.append(game_num + 1)
            win_rates.append(results['win_rate'])
            draw_rates.append(results['draws'] / eval_games * 100)
            print(f"After {game_num + 1:>5} games vs MCTS: "
                  f"Q win rate={results['win_rate']:5.1f}%  draws={results['draws']:>2}/{eval_games}  "
                  f"(Q-table size={len(agent.q_table)})")

    return agent, game_log, checkpoints, win_rates, draw_rates


def plot_dashboard(game_log, checkpoints, win_rates, draw_rates, save_path='dashboard.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Win rate vs MCTS over training (the headline "gap closing" plot)
    ax = axes[0, 0]
    ax.plot(checkpoints, win_rates, marker='o', color='tab:blue', label='Q-agent win rate vs MCTS')
    ax.plot(checkpoints, draw_rates, marker='s', color='tab:gray', label='Draw rate')
    ax.axhline(50, color='black', linestyle=':', linewidth=1, label='50% (even match)')
    ax.set_xlabel('Training games played')
    ax.set_ylabel('% of evaluation games')
    ax.set_title('Closing the gap: Q-agent vs MCTS')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Outcome per training game, smoothed (rolling win/draw/loss rate)
    ax = axes[0, 1]
    window = 50
    outcomes = [g['outcome'] for g in game_log]
    xs = list(range(1, len(outcomes) + 1))
    win_flags = [1 if o == 'q_win' else 0 for o in outcomes]
    draw_flags = [1 if o == 'draw' else 0 for o in outcomes]
    rolling_win = _rolling_mean(win_flags, window)
    rolling_draw = _rolling_mean(draw_flags, window)
    ax.plot(xs, [w * 100 for w in rolling_win], color='tab:blue', label=f'Win rate ({window}-game rolling)')
    ax.plot(xs, [d * 100 for d in rolling_draw], color='tab:gray', label=f'Draw rate ({window}-game rolling)')
    ax.set_xlabel('Training game number')
    ax.set_ylabel('%')
    ax.set_title('Training-time outcomes (with exploration on)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Exploration rate decay
    ax = axes[1, 0]
    ax.plot(xs, [g['exploration_rate'] for g in game_log], color='tab:orange')
    ax.set_xlabel('Training game number')
    ax.set_ylabel('Exploration rate (epsilon)')
    ax.set_title('Exploration decay')
    ax.grid(True, alpha=0.3)

    # 4. Game length distribution
    ax = axes[1, 1]
    lengths = [g['move_count'] for g in game_log]
    ax.hist(lengths, bins=20, color='tab:purple', alpha=0.8)
    ax.set_xlabel('Total moves in game')
    ax.set_ylabel('Number of games')
    ax.set_title('Game length distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nSaved dashboard to {save_path}")
    plt.show()


def _rolling_mean(values, window):
    result = []
    running_sum = 0
    for i, v in enumerate(values):
        running_sum += v
        if i >= window:
            running_sum -= values[i - window]
        result.append(running_sum / min(i + 1, window))
    return result


if __name__ == "__main__":
    agent, game_log, checkpoints, win_rates, draw_rates = train_vs_mcts()

    with open('game_log.json', 'w') as f:
        json.dump(game_log, f, indent=2)
    print(f"\nSaved full game-by-game log to game_log.json ({len(game_log)} games)")

    agent.save_model('q_vs_mcts_model.pkl')
    plot_dashboard(game_log, checkpoints, win_rates, draw_rates)
