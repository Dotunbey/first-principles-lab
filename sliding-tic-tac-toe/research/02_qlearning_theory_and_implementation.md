# Q-Learning: Theory and Implementation

## Core vocabulary

- **State (s)**: the board layout right now (plus, in this game, which phase - placement
  or sliding - and whose turn it is; see "state representation" below).
- **Action (a)**: a legal move from that state.
- **Policy**: the rule used to pick an action given a state - a Q-table becomes a policy
  once you always pick the highest-valued action.
- **Reward (r)**: the number received for a transition.
- **Episode**: one complete game, start to finish.
- **Q-value, Q(s,a)**: an estimate of "if I take action `a` in state `s`, then play well
  from there on, how much total (discounted) reward will I eventually get" - not just the
  immediate reward, the whole future.

## The Bellman update, term by term

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ]
```

- `Q(s,a)` - current estimate, before this update.
- `alpha` (learning rate) - how much to trust this one new data point vs. everything
  learned before.
- `r` - reward just received.
- `s'` - the state landed in after the move.
- `max Q(s',a')` - the best value achievable from the next state - this is what lets
  reward from a win at the last move flow backward to influence a much earlier move.
- `gamma` (discount factor) - how much future reward matters relative to immediate reward.
- The bracketed term is the **TD error**: the gap between what was expected and what was
  just observed.

### Worked numeric example (done by hand before any code existed)

Given alpha=0.1, gamma=0.9:

A winning final move: `Q(s,a)` was 0.2, reward r=1 (win), no future state to look ahead
into:

```
Q(s,a) <- 0.2 + 0.1 x (1 + 0.9x0 - 0.2) = 0.2 + 0.1x0.8 = 0.28
```

The move *before* that one (which set up the win), where `Q(s,a)` was 0.05 and the
"future value" is the 0.28 just computed:

```
Q(s,a) <- 0.05 + 0.1 x (0 + 0.9x0.28 - 0.05) = 0.05 + 0.1x0.202 = 0.0702
```

This is the whole mechanism: value trickles backward from the terminal reward, one game at
a time, one step of credit assignment per move.

## Exploration vs. exploitation

Always picking the highest-Q action risks getting stuck on a mediocre strategy discovered
early. **Epsilon-greedy**: with probability epsilon, play a random legal move (explore);
otherwise play the best-known move (exploit). Epsilon is annealed down over training - high
early (the table is empty/unreliable, so explore heavily), lower later (trust the
increasingly reliable table more).

## State representation actually used

`hash_state(game)` returns `f"{tuple(game.board)}_{game.phase}_{game.current_player}"` -
board contents, which phase (placement/sliding), and whose turn it is. Deliberately
**excluded**: move history / how many times this exact position has recurred (needed by the
engine's own threefold-repetition rule). The agent judges a position on its own merits, not
on how it got there - repetition-avoidance is left entirely to the environment's own draw
rule, not encoded into the state the agent reasons about.

## Implementation (`q_learning.py`, `QLearningAgent`)

- `choose_action(game, greedy=False)` - epsilon-greedy move selection; `greedy=True` is
  used at evaluation time (no exploration).
- `train_from_self_play(num_games, epsilon_start, epsilon_end)` - **true self-play**: the
  *same* agent and Q-table drive both X and O (not a fixed opponent, and not "the agent vs.
  random"), with epsilon decaying linearly across the whole run. This differs from an
  earlier version of the same idea (used in the precursor Oware project) where self-play
  was implemented against a purely random opponent - a weaker training signal, since
  learning to beat randomness doesn't teach real defensive play.
- `update_q_values(history, winner, player)` - the backward pass. Reward scheme is
  deliberately the **plain textbook version**: only the terminal outcome (win=+1, loss=-1,
  draw=0), nothing shaped in along the way. Every non-terminal move earns credit purely
  through backward discounting, not through any hand-designed intermediate bonus. This is
  the same update-formula shape used in Sutton & Barto's canonical tic-tac-toe example, and
  was chosen deliberately as a "ground truth" baseline to compare a later, shaped version
  against (see `06_reward_shaping_experiment.md`).
- `save_model` / `load_model` - persist the table (a `defaultdict` of `defaultdict`, whose
  `lambda` default factory isn't directly picklable, so it's flattened to plain dicts
  first).
- `evaluate_vs_random(agent, num_games)` - plays the trained agent (greedy, X) against a
  uniformly random opponent (O), used as a strength benchmark.

## Validation run

Training for 5,000 self-play games (epsilon 0.3 -> 0.05):

```
Completed 500 games  (exploration_rate=0.275, Q-table size=2611)
Completed 1000 games (exploration_rate=0.250, Q-table size=3450)
Completed 1500 games (exploration_rate=0.225, Q-table size=3797)
Completed 2000 games (exploration_rate=0.200, Q-table size=3978)
Completed 2500 games (exploration_rate=0.175, Q-table size=4108)
Completed 3000 games (exploration_rate=0.150, Q-table size=4239)
Completed 3500 games (exploration_rate=0.125, Q-table size=4313)
Completed 4000 games (exploration_rate=0.100, Q-table size=4357)
Completed 4500 games (exploration_rate=0.075, Q-table size=4398)
Completed 5000 games (exploration_rate=0.050, Q-table size=4424)

Final: wins_X=2691, wins_O=1720, draws=589, q_table_size=4424
Win rate vs random opponent (200 games, greedy): 199/200 = 99.5%
```

The final Q-table size (4,424) closely matches the by-hand combinatorial prediction
(~4,030-4,974, see `01_game_definition_and_state_space.md`) - an independent confirmation,
arrived at from a completely different direction (empirical training vs. combinatorial
counting), that the earlier state-space math was correct.
