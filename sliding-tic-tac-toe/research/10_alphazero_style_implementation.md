# Closing the Gap: A Mini-AlphaZero, Built From Scratch

## Motivation

After watching DeepMind's AlphaZero explainer video and mapping its ideas onto everything
already built here, five concrete gaps were identified against this project's
table-based approach (see the conversation record for the full comparison): table vs.
network, blind vs. guided MCTS, separate components vs. one integrated loop, no ground
truth vs. our exact solver (an advantage, not a gap), and shallow vs. deep self-play. This
document covers actually closing the first three, built in three stages, each validated
before moving to the next - consistent with how every other component in this project
was built and tested.

Implemented entirely in plain Python + numpy, no ML framework, matching how every other
piece of this project (Q-learning, MCTS, the solver, TD(0)) was built from first
principles rather than using a library.

## Stage 1: the network (`neural_net.py`)

A small hand-written neural network (2 hidden layers, ReLU, He initialization) with two
output heads sharing the trunk:
- **Value head**: one number in [-1, 1] (tanh) - "how good is this position for whoever is
  about to move," directly comparable to the solver's own `negamax_value` convention.
- **Policy head**: 90 logits (9 placement actions + 81 from/to sliding combinations),
  masked to legal moves whenever actually used.

Input encoding follows AlphaZero's own "canonicalization" trick: always encode the board
from the *current player's* perspective (my pieces / opponent's pieces / phase), so the
network never has to separately learn "I am X" vs. "I am O" as different concepts.

Forward pass, backward pass (hand-derived gradients for both heads), and the Adam
optimizer were all written manually - no `torch`/`tensorflow`.

### Validating the core claim: does it actually generalize?

The one thing a table can never do is say anything useful about a state it has never
exactly visited. Trained the value head alone on 80% of the solver's 4,974 exactly-solved
states, tested on the 20% held out (never seen during training):

```
epoch  1: test MAE (unseen states) = 0.566  (~random guessing)
epoch 60: test MAE (unseen states) = 0.206  (real, substantial accuracy)
```

This is the concrete, measured proof that this approach closes the fundamental gap: the
network reaches meaningfully accurate estimates on positions it was never trained on,
purely from similarity to positions it has seen - something no amount of additional
training could ever give the Q-table approach, which has zero information about an
unvisited state by construction.

## Stage 2: guided MCTS (`neural_mcts.py`)

Replaces `mcts.py`'s plain UCB1 selection and rollout-based leaf evaluation with:
- **PUCT selection**: `Q(s,a) + c_puct * prior * sqrt(N_parent) / (1 + N(s,a))`, where the
  policy network's prior biases which branches get explored, instead of every move
  starting out equally interesting.
- **Instant value-network leaf evaluation**: no rollout is ever played out; the value
  network's output at a newly-expanded leaf is used directly.
- Values flow back up the tree with a sign flip at every level (the same negamax
  convention used everywhere else in this project - the solver, plain MCTS's
  backpropagation, and TD(0)'s bootstrap all do the same thing).

### A real bug found and fixed during validation

First validation attempt was dramatically wrong: pitted against a random opponent (with
only the value head trained, policy head still random), it lost 14/20 games and never won
once - worse than random play, which should be essentially impossible for any functioning
search.

Root cause: `game_engine.py`'s move-application methods `return` early on a winning move,
*before* flipping `current_player` - so immediately after a win, `current_player` is still
the winner, not "whoever's turn would be next." The terminal-value function had assumed
that field always flips, silently inverting every win/loss it encountered.

Notably, `perfect_solver.py` had already avoided this exact trap - its state enumeration
compares against the pre-move mover rather than trusting the post-move `current_player`
field - but that defensive pattern wasn't carried over when writing the new terminal-value
check. Fixed by tracking `player_just_moved` explicitly at node-creation time (the same
approach `mcts.py`'s original `TreeNode` already used), rather than trusting the engine's
post-terminal state.

After the fix, re-validated on a hand-constructed one-move-win scenario (confirmed the
search now correctly finds and heavily exploits the winning move: 193/200 visits), then
re-ran the full random-opponent test:

```
Before fix: 0 wins, 14 losses, 6 draws (worse than random)
After fix:  20 wins, 0 losses, 0 draws
```

Also markedly faster than the old rollout-based MCTS (2.4s vs. 11.8s for 20 games) - the
value network's instant evaluation replaces an entire simulated game per leaf.

This was achieved with an untrained (random) policy head - all of the improvement here
came from the value network replacing rollouts, not from search guidance yet.

## Stage 3: the closed self-play loop (`alphazero_selfplay.py`)

Ties the previous two stages together into the actual AlphaZero-style cycle:
1. Generate a self-play game where every move comes from `neural_mcts_search`.
2. Record each visited position's input encoding and the search's own visit-count
   distribution (used as the policy training target - literally "what did more thinking
   discover was better than the network's raw guess").
3. Once the game ends, back-fill every recorded position with the real final outcome
   (from that position's own mover's perspective) as its value target - a Monte
   Carlo-style target, which (per the earlier research conversation) is actually closer to
   how AlphaZero's own value network is trained than our own `td0` mode is.
4. Train the network on the accumulated batch of examples.
5. Repeat - a stronger network makes the next round of self-play stronger, which produces
   better training data, which strengthens the network further.

A small smoke test (5 iterations, 10 games each, from a freshly-initialized random
network) confirmed the mechanics run cleanly end to end - real, varied game outcomes, and
a value-error-vs-solver metric (the network's own version of this project's core "training
loss") trending in the 0.63-0.72 range this early, expected given the tiny scale of the
smoke test and a value head now training jointly with a not-yet-useful policy head, rather
than the isolated, warm-started value-only test from Stage 1.

## The real training run: 300 iterations, 20 games each

A full run followed the smoke test: 300 self-play "generations," 20 games per iteration
(6,000 games total), 100 MCTS simulations per move, ~33 minutes total wall-clock time. A
small read-only monitoring dashboard (`selfplay_dashboard/`, port 5004) was built alongside
it purely to watch the log file live, without touching the training process at all.

```
First 20 iterations avg value_error_vs_solver: 0.738
Last  20 iterations avg value_error_vs_solver: 0.544
```

**Real, if incomplete, improvement.** The network's value estimates measurably closed the
gap to the exact solver over the course of training - not down to the ~0.206 the isolated,
fully-supervised Stage 1 test reached (expected: that test trained directly on all 4,974
exact answers at once, while self-play only ever sees the noisy, partial signal of games
it actually plays), but a genuine, real reduction over 6,000 games of self-generated
experience with no outside answers provided.

## A second finding: a real X/O skill asymmetry, not exploration noise

The win/draw distribution across the run showed an increasingly one-sided pattern:

```
First 20 iterations avg: wins_X=9.1, wins_O=7.35, draws=3.55  (roughly balanced)
Last  20 iterations avg: wins_X=17.2, wins_O=1.4, draws=1.4   (X dominant)
```

Since self-play always samples moves from the search's visit distribution with some
temperature (more randomness early in a game, for exploration diversity), the natural
question was whether O's collapsing win rate was just an artifact of exploration noise
rather than a real defensive weakness. Tested directly: loaded the final trained network
and played 20 games with **both sides fully greedy** (temperature effectively 0, no
Dirichlet noise, no sampling - the single most-visited move every time, for both players).

```
Result: X wins 20/20, O wins 0/20, draws 0/20
```

Removing all randomness made the asymmetry *more* extreme, not less - ruling out
exploration noise as the explanation. **This is a genuine, learned imbalance in the shared
network itself**: it has learned to attack as X (the side with a real, provable
first-move advantage - recall X can force a win under perfect play) far better than it has
learned to defend as O (whose best possible true outcome is a draw). A plausible mechanism:
since X's real advantage means self-play naturally produces more decisive X wins early on,
the shared network receives disproportionately more "how to attack as X" training examples
than "how to defend as O" ones, and that imbalance can compound - a network that defends
poorly as O keeps losing as O, generating even more lopsided data. This is structurally
similar in spirit to the "drawing trap" found in the TD(0) experiments
(`08_td0_online_learning.md`), but inverted: there, insufficient exploration led to
*too much* safety; here, a real structural asymmetry in the game itself appears to be
under-represented in the correction the O-side needs.

## Status

All three stages are built, validated, and have now been run at real scale. Value error
vs. the exact solver shows genuine improvement (0.738 -> 0.544) but has not converged
anywhere near the Stage 1 ceiling (0.206), and a real, confirmed X/O skill asymmetry has
emerged that plain additional training did not self-correct within this budget. Natural
next steps: a longer run to see whether the asymmetry eventually corrects itself as more
O-losses accumulate real training signal, or a more direct fix such as deliberately
balancing training examples between X-to-move and O-to-move positions, mirroring how
`06_reward_shaping_experiment.md` and `08_td0_online_learning.md` each needed a targeted
diagnosis (not just "more training") to move past their own respective plateaus.
