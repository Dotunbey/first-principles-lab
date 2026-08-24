# Oware Learning AI

A Python implementation of the Oware board game (also known as Awari) paired with a Q-learning agent that learns to play through self-play and expert-game replay.

## Overview

`oware_engine.py` implements the game rules (board, valid moves, captures, game-over detection). `enhanced_learn_oware.py` implements a tabular Q-learning agent that trains against itself and against recorded expert games, then persists the learned Q-table to disk.

## Components

1. **Game engine** (`oware_engine.py`) — board state, move validation, capture rules, win/draw detection.
2. **Q-learning agent** (`enhanced_learn_oware.py`) — epsilon-greedy action selection over a `state -> action -> Q-value` table, with exploration decaying over training.
3. **Reward shaping** (`reward_system.py`) — documents the reward structure (win/loss/draw outcome plus capture and center-house-control bonuses); these bonuses are wired into `EnhancedOwareLearningAI.compute_step_reward` and folded into the backward Q-value update.
4. **Expert game database** (`oware_games_database.json`) — recorded move sequences from a starting board, replayed through the engine during `train_from_expert_games` to produce real state-action-reward transitions.
5. **Visualization** (`visualize_oware.py`, `animate_oware.py`, `final_visualization.py`) — inspecting board states and training progress.

## Usage

```bash
pip install -r requirements.txt
python enhanced_learn_oware.py
```

This trains an agent via self-play (both sides driven by the same Q-table, exploration rate decaying from `epsilon_start` to `epsilon_end`), then trains further on the recorded games in `oware_games_database.json`, and saves the result to `oware_model.pkl`.

```python
from enhanced_learn_oware import EnhancedOwareLearningAI

ai = EnhancedOwareLearningAI()
ai.train_from_self_play(num_games=500)
ai.save_model('oware_model.pkl')
```

## Known limitations

- **Tabular state representation**: states are hashed by exact board+score+player, so the agent doesn't generalize across similar-but-distinct positions. A large number of self-play games is needed to cover the state space, and most states are visited only once.
- **No search/lookahead**: move selection is purely greedy over learned Q-values (no minimax or MCTS), so play quality is bounded by what tabular Q-learning alone can discover.
- **Small expert dataset**: `oware_games_database.json` contains a handful of illustrative games, not a large corpus of real tournament records.

## Future Enhancements

- Feature-based (linear or neural) function approximation instead of a raw state-hash table, to generalize across positions.
- Minimax/alpha-beta or MCTS move selection layered on top of the learned values.
- A larger, real expert-game corpus.
