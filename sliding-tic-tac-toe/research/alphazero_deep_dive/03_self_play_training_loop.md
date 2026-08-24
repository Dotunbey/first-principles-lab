# The Self-Play Training Loop: Closing the Circle

Code: `alphazero_selfplay.py`. Read `01_neural_network_architecture.md` and
`02_mcts_with_puct.md` first - this document assumes both the network and the guided
search already exist and work.

## 1. The big idea in one sentence

Play games using the network-guided search; use what the search discovered (and what
actually happened by the end of the game) as new training examples for the network;
repeat with the improved network - each cycle should make both the search and the network
a little bit stronger than the last.

```
      ┌─────────────────────────────────────────────────────────────┐
      │                                                             │
      ▼                                                             │
┌──────────────┐        ┌──────────────┐        ┌──────────────┐   │
│  SELF-PLAY    │        │   RECORD      │        │   TRAIN       │   │
│  (guided MCTS │──────▶│  every position│──────▶│  the network   │───┘
│  plays a full  │        │  visited, plus │        │  on this batch │
│  game)         │        │  the outcome   │        │                │
└──────────────┘        └──────────────┘        └──────────────┘
     uses the                                        produces a
   CURRENT network                                  BETTER network
                                                     for next time
```

This is genuinely different from every other approach tried earlier in this project
(sparse Q-learning, potential-shaped Q-learning, TD(0)) in one crucial respect: those all
trained a single component in isolation, playing against a *fixed* opponent (MCTS, the
perfect solver, or a copy of itself with no search at all). Here, the network and the
search it powers are locked together in a continuously improving loop - a stronger network
makes the guided search stronger, and a stronger search produces better training examples,
which makes the network stronger still.

## 2. Generating one self-play game, step by step

```python
def play_one_selfplay_game(net, iterations=100, temperature_moves=6):
    game = SlidingTicTacToe()
    examples = []
    move_number = 0

    while not game.game_over:
        move, root = neural_mcts_search(game, net, iterations=iterations, add_noise=True)

        temperature = 1.0 if move_number < temperature_moves else 0.1
        policy_target = visit_count_policy(root, temperature=temperature)

        x = encode_state(game.board, game.phase, game.current_player)
        examples.append((x, policy_target, game.current_player))

        if temperature > 0.5:
            # sample from the (temperature-adjusted) visit distribution
            move = <sampled from policy_target, restricted to legal moves>

        game.make_move(move)
        move_number += 1
    ...
```

For every single move of the game:
1. Run a full guided MCTS search from the current position (with Dirichlet noise mixed
   into the root's priors, since this is self-play - see `02_mcts_with_puct.md` section 8).
2. Convert the search's final visit counts into a probability distribution
   (`visit_count_policy`) - this distribution is exactly what gets used as this position's
   **policy training target**. Read that sentence twice: the training target for "what
   should the network's policy head have said" isn't a human label or a fixed rule - it's
   literally "what did a few hundred simulations of lookahead conclude was best," which is
   almost always a better answer than the network's own untrained guess. This is the core
   mechanism that makes self-play *improve* the network rather than just reproduce it.
3. Record the current position (encoded as the 19-number input vector) alongside that
   policy target.
4. Actually make a move - sampled from the (temperature-adjusted) distribution early in
   the game for diversity, or effectively the single best move later on.

## 3. Filling in the value target: why it has to wait until the game ends

Unlike the policy target (known immediately, from that turn's own search), the **value**
target - "was this position actually good or bad?" - can't be known until the game is
completely over. So every recorded position is back-filled once the winner is known:

```python
winner = game.winner
for x, policy_target, mover in examples:
    if winner == -1:
        value_target = 0.0
    else:
        value_target = 1.0 if winner == mover else -1.0
```

In plain language: every position gets graded, after the fact, by whether *that specific
position's own mover* ended up winning, losing, or drawing the game.

This detail matters and connects back to a point made in the earlier, table-based part of
this project (`../06_reward_shaping_experiment.md`): using the **actual final outcome** as
the training target for every visited position (rather than a step-by-step bootstrapped
estimate, the way this project's `td0` mode does) is a Monte Carlo-style target, not a
Temporal-Difference one. This is worth noting explicitly because it means the "pure,
solver-free, every-move-matters" idea explored with `td0` earlier in this project is
actually a genuine *departure* from how the real AlphaZero itself assigns credit, not a
reproduction of it - AlphaZero's own value network is trained exactly this same
Monte-Carlo way.

```
Diagram: one finished game, and how each visited position gets graded

  Move 1        Move 2        Move 3        Move 4 (winning move)
 (X's turn)    (O's turn)    (X's turn)     (X's turn)
     │             │             │              │
     ▼             ▼             ▼              ▼
 recorded       recorded      recorded       recorded
     │             │             │              │
     └─────────────┴─────────────┴──────────────┘
                        │
                 GAME ENDS: X wins
                        │
     ┌─────────────┬─────────────┬──────────────┐
     ▼             ▼             ▼              ▼
 value = +1    value = -1    value = +1     value = +1
 (X's move,   (O's move,    (X's move,     (X's move,
  X won)       X won = bad   X won)         X won)
               for O)
```

## 4. Training on the accumulated examples

```python
def train_on_examples(net, examples, batch_size=64, learning_rate=0.001):
    random.shuffle(examples)
    for each batch of 64 examples:
        x_batch, policy_batch, value_batch = <unpack>
        legal_mask = (policy_batch > 0)
        net.train_step(x_batch, value_batch, policy_batch, legal_mask, learning_rate)
```

Straightforward mini-batch training, using the backpropagation math from
`01_neural_network_architecture.md` section 5, applied to whatever pool of examples is
available (see the replay buffer discussion below for what "available" means in practice).

## 5. The full outer loop: generations

```
for iteration in range(NUM_ITERATIONS):
    # 1. Generate a batch of fresh self-play games with the CURRENT network
    for _ in range(GAMES_PER_ITERATION):
        examples, winner = play_one_selfplay_game(net, iterations=MCTS_ITERS)
        replay_buffer.extend(examples)

    # 2. Train the network on the accumulated examples
    train_on_examples(net, list(replay_buffer))

    # 3. Measure progress against the exact solver (this project's unique advantage)
    err = value_error_vs_solver(net, solved_states)

    # 4. Checkpoint periodically
    if iteration % CHECKPOINT_EVERY == 0:
        net.save(...)
```

Each pass through this loop is called a "generation" - self-play with the current network,
then a training step that (hopefully) produces a slightly better network for the next
generation's self-play.

## 6. The replay buffer: why not just train on the newest games?

An early design question: should training use only the games just generated this
iteration, or a larger pool spanning several recent iterations? This project uses a
**replay buffer** - a fixed-capacity rolling window (8,000 examples in the "balanced"
run) that keeps recent games and discards the oldest ones as new games come in:

```
Iteration 1:  [games 1-20]                                    → buffer: 20 games' worth
Iteration 2:  [games 1-20][games 21-40]                       → buffer: 40 games' worth
   ...
Iteration N:  [   ...    ][games from the last several iterations, oldest dropped   ]
```

The reasoning: training on only the freshest 20 games each time means each training step
sees a small, noisy sample, and any imbalance in that specific small sample (see
`04_bugs_lessons_and_limitations.md` for a concrete case where this mattered a great deal)
directly shapes that update with no averaging-out effect. A larger rolling buffer smooths
this out and lets rarer example types get reused across several training steps instead of
being discarded after a single pass.

## 7. Measuring progress: the network's own "training loss" against ground truth

Because this project's game is small enough to solve exactly (`../04`), progress can be
tracked against the actual correct answer, not just an indirect proxy like win rate:

```python
def value_error_vs_solver(net, sample_states, sample_size=300):
    sample some real solved positions
    for each: compare net.predict(...) against the solver's exact negamax_value(...)
    return the average |difference|
```

This is the network's own version of the `compute_value_error` metric already used
throughout the table-based parts of this project (`../07_summary_and_open_questions.md`)
- an exact, zero-sampling-noise measurement of how close the network's beliefs are to the
true, solved values, entirely independent of which opponent it's currently facing.

## What's next

`04_bugs_lessons_and_limitations.md` walks through, as full case studies, the two real
bugs this build hit (one fixed cleanly, one only partially fixed) and what each one
teaches about building a system like this correctly.
