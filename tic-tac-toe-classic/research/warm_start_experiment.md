# Does warm-starting from the sliding-game agent help? No - it actively hurts.

## The idea

The sliding-tic-tac-toe project already has a heavily-trained Q-learning agent
(`../../sliding-tic-tac-toe/web_dashboard/live_session.pkl`, 48,000+ games). Its
placement phase - each side places 3 pieces onto an empty 3x3 board, one at a
time, before switching to sliding - plays out on an IDENTICAL board
representation to classic tic-tac-toe's entire game, for as long as it lasts.
Since `QLearningAgent.hash_state` produces `f"{board}_{phase}_{player}"`, and
this project's `game_engine.py` keeps `phase` fixed at the literal string
`'placement'` for its whole game specifically to match that format, the
sliding agent's placement-phase state hashes come out BYTE-IDENTICAL to
classic tic-tac-toe's early-game states.

The natural hypothesis: bootstrap a classic agent's Q-table from those
matching entries (`bootstrap_from_sliding_agent.py`) instead of starting from
zero, and it should have a head start on the opening.

## The experiment

Two identically-configured dashboards, same hyperparameters (`learning_rate`
0.1, `discount_factor` 0.9, `epsilon_start` 0.4 -> `epsilon_end` 0.05 over 300
games, `opponent='mcts'` at 150 iterations, sparse Monte Carlo reward, no
shaping):

- **5013 (warm-started)**: Q-table pre-loaded from 826 placement-phase states
  extracted from the sliding agent's session (970 sliding-phase states
  discarded - no classic-game equivalent).
- **5014 (fresh)**: empty Q-table, learns everything from real classic games.

Trained side by side, `value_error` (mean `|agent's best-known value - exact
classic-solver value|`, the same zero-noise ground-truth metric used
throughout the sliding project) checked at matching game counts.

## The result

```
game_num | warm-started (5013) | fresh (5014)
    25    |       0.5624        |    0.4275   (fresh better by 0.135)
   300    |       0.5370        |    0.4840   (fresh better by 0.053)
  1000    |       0.5265        |    0.4753   (fresh better by 0.051)
  2000    |       0.5253        |    0.4595   (fresh better by 0.066)
  3400    |       0.5219        |    0.4609   (fresh better by 0.061)
```

Fresh was ahead at every single checkpoint compared, from the very first one.
The warm-started run was stopped at game 9,046 with `value_error` still
around 0.52 - never having closed the gap that was already present at game 25.

## Why: verifying it wasn't just "needs more time"

Before concluding the warm start actively hurt (rather than "helped less than
hoped"), the inherited Q-values were checked against the CLASSIC solver's
ground truth directly, at the moment they were loaded - game 0, before a
single classic game had been played:

```python
# mean |inherited Q-value - classic ground truth|, all 826 warm-started states
0.5686
```

This is WORSE than the fresh agent's value_error at its very first checkpoint
(0.4275, game 25) - the warm start didn't just fail to provide a head start,
it started from a position further from the truth than 25 real games of
learning from scratch produced. The worst individual mismatches were nearly
maximally wrong - one position:

```
board=[1, 1, 0, 1, 2, 0, 0, 0, 2]  player=O
inherited value (from sliding-game training): +0.863  (near-certain win)
true classic value:                           -1.000  (certain loss)
```

## The mechanism: identical boards, different games, different values

The reasoning that made the transfer seem sound has exactly one flaw, and
it's not in the state-hash matching (that part is genuinely correct - the
boards really are byte-identical for the first six plies). The flaw is in
what a Q-value actually represents: not "whose marks are where," but "the
expected outcome given what happens next from here."

- In the **sliding game**, a placement-phase Q-value encodes "assuming this
  eventually transitions to sliding-phase maneuvering" - after both sides
  place 3 pieces, the game becomes about mobility and which piece can slide
  into which line.
- In **classic tic-tac-toe**, the same board's Q-value should encode
  "assuming direct placement continues until the board fills" - up to 9
  total marks (5 X, 4 O), no sliding ever.

Those are different games from the placement phase onward, even though nothing
about the board itself reveals that difference. A move that sets up a strong
future slide may be a mediocre (or actively bad) move if the actual next 3-6
plies are more placements instead. The board being identical was a
coincidence of the opening; the strategy the position is being judged
against diverges immediately after it.

## Verdict

**Do not use this bootstrap approach as the default way to initialize a
classic tic-tac-toe agent.** It is kept in the codebase
(`bootstrap_from_sliding_agent.py`) as a documented negative result, not a
recommended starting point. A fresh, from-scratch agent reaches better
value-error at every game count tested, likely because it never has to spend
real games first UN-learning sliding-game-specific assumptions before making
net-new progress.

This is also a useful general lesson for any future cross-game or
cross-task transfer-learning attempt in this whole project: matching state
REPRESENTATIONS (same hash format, same board encoding) is necessary but not
sufficient for a value function to transfer - what determines whether a
value is actually valid is what game/task continues from that state, which
can differ even when the state itself looks the same.
