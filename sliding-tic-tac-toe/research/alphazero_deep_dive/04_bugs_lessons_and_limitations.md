# Bugs, Lessons, and an Honest Limitation

Read `01`, `02`, and `03` first. This document is a full case study of two real problems
found while building this system - one bug that was cleanly fixed, and one fix attempt
that only partially worked, with the reasoning for *why* it fell short.

## Bug 1: the terminal-value sign inversion

### The symptom, before any code was inspected

The very first end-to-end test of guided MCTS (Stage 2, using a network whose value head
had already been validated to generalize reasonably well) was run against a random
opponent - the same sanity check plain MCTS passed at 20/20 wins early in this project
(`../03_mcts_theory_and_implementation.md`).

Guided MCTS **lost 14 games out of 20**, drew the remaining 6, and won **zero**.

This number alone was the whole diagnostic clue needed: a search that's even mildly
functional, paired with *any* nonzero information about position quality, should struggle
to lose to fully random play, let alone lose the *majority* of games. Something wasn't
"a bit weak" - something was fundamentally inverted.

### Finding it: a controlled, minimal test case

Rather than stare at the full search code, the standard debugging move used throughout
this whole project (see `../07_summary_and_open_questions.md`'s "recurring methodological
pattern" section) was applied again: construct the smallest possible scenario where the
right answer is obviously known, and check whether the code agrees.

```
Board:                    Legal moves for X: {2, 5, 6, 7, 8}
 X  X  .                  Move 2 completes the top row -> INSTANT WIN for X.
 O  O  .                  Every other move does not win immediately.
 .  .  .
```

Running guided MCTS on this exact position (200 simulations) produced:

```
move 2: visits=1,   value=1.000   <- correctly identified as a WIN...
move 6: visits=116, value=0.024   <- ...but barely revisited, and a mediocre
                                      move was chosen 116 times more often!
```

The search *did* correctly discover that move 2 wins outright (value = 1.000, exactly
right) - but then never went back to confirm and exploit it. That's the signature of a
sign error in the selection formula: a move correctly identified as excellent was being
scored as if it were terrible, so the search kept avoiding it.

### Root cause

`game_engine.py`'s move-application methods **return immediately upon detecting a win**,
before the line that normally flips whose turn it is:

```python
def _make_placement_move(self, index):
    ...
    if self._check_win(index):
        return              # <-- exits HERE. current_player is NEVER flipped on a win.
    ...
    self.current_player = O if self.current_player == X else X   # only reached if no win
```

So immediately after a winning move, `game.current_player` still names the **winner**, not
"whoever's turn would be next" - which is what every other part of this system (and every
non-terminal state) assumes that field means.

The very first version of the terminal-value function trusted that field directly:

```python
# WRONG - assumes current_player always means "next to move," which is false right after a win
def _terminal_value(game_state):
    if game_state.winner == -1:
        return 0.0
    return 1.0 if game_state.winner == game_state.current_player else -1.0
```

Since `current_player` after a win is actually the *winner*, this function was
accidentally computing "is the winner... the winner?" - which is always true - so it
always returned `+1.0`, **regardless of who actually won**. From the perspective of
whichever node in the tree was on the *losing* side of that terminal state, a `+1.0` value
then got sign-flipped on the way up (per the negamax convention, `02_mcts_with_puct.md`
section 6) into a `-1.0` - inverting the outcome for exactly the branch of the tree that
most needed the correct signal.

Interestingly, `perfect_solver.py`, built much earlier in this project, had **already
sidestepped this exact trap** - its state enumeration explicitly captures the mover's
identity *before* applying a move and compares against that, never trusting the post-move
`current_player` field for a terminal position. That defensive habit simply wasn't carried
over when this new code was written from scratch.

### The fix

Track **who actually moved to create this node** as an explicit, permanent property of
the node itself - set once, at creation time, and never re-derived from a field that turns
out to be unreliable exactly when it matters most:

```python
class NeuralNode:
    def __init__(self, ..., player_just_moved=None):
        self.player_just_moved = player_just_moved   # captured BEFORE the move was made

def _terminal_value(node):
    mover = node.player_just_moved
    if node.game_state.winner == -1:
        value_to_mover = 0.0
    elif node.game_state.winner == mover:
        value_to_mover = 1.0
    else:
        value_to_mover = -1.0
    return -value_to_mover   # convert to "value to whoever's next" convention, for consistency
```

This is exactly the same defensive pattern the original plain `mcts.py`'s `TreeNode`
already used (`player_just_moved`, set at expansion time) - the fix effectively brought
the new code in line with a lesson the project had already learned once, just not yet
propagated to this new file.

### Verification after the fix

Same minimal test case:

```
move 2: visits=193, value=-1.000   <- now heavily exploited (the -1.000 shown here is the
                                       INTERNAL bookkeeping value, "value to next mover" -
                                       which for a finished game is moot, but selection
                                       correctly negates it again into +1.000 for X's own
                                       decision, hence 193/200 visits piling onto it)
```

And the full random-opponent test:

```
Before fix: 0 wins, 14 losses,  6 draws   (worse than random)
After fix: 20 wins,  0 losses,  0 draws
```

A single sign convention, wrong at exactly one place, was the entire difference between
"loses to random play" and "wins every single game."

## Bug/Limitation 2: the X/O skill asymmetry, and a fix that only partially worked

### The symptom

Over a full 300-iteration self-play training run, the win/draw distribution grew
increasingly lopsided:

```
First 20 iterations, average per iteration:  X wins 9.1, O wins 7.35, draws 3.55  (balanced)
Last  20 iterations, average per iteration:  X wins 17.2, O wins 1.4, draws 1.4   (X-dominant)
```

### Ruling out the obvious alternative explanation first

Self-play moves are sampled with some randomness (temperature, Dirichlet noise - see
`02_mcts_with_puct.md` section 8), so the first hypothesis worth checking was: is this
just exploration noise, not a real skill gap? Tested directly by loading the trained
network and replaying 20 games with **all** randomness removed - both sides playing the
single most-visited move, deterministically, every time:

```
Result: X wins 20/20, O wins 0/20, draws 0/20   <- MORE one-sided than with exploration, not less
```

This rules out exploration noise as the explanation. The imbalance is a genuine property
of what the shared network has learned.

### Why this happens: a plausible, compounding mechanism

This game has a real, structural asymmetry: `../04_perfect_solver_exact_solution.md`
proved **X can force a win** with perfect play, while O's best possible outcome is a draw.
Self-play therefore *naturally* generates more decisive X-favoring outcomes than
O-favoring ones, even before any bug or design flaw is involved - and since the *same*
network controls both sides, that lopsided data feeds disproportionately more "how to
attack as X" signal into the shared weights than "how to defend as O" signal.

```
     more X wins  →  more "X attacking" training examples
           ▲                          │
           │                          ▼
   O keeps losing  ←  network gets even better at X's attack,
                       no matching improvement at O's defense
```

A self-reinforcing loop, structurally similar in spirit (though a different underlying
cause) to the "drawing trap" found earlier in this project's TD(0) experiments
(`../08_td0_online_learning.md`) - there, insufficient exploration caused *too much*
safety; here, a real asymmetry in the game itself compounds because both sides share one
learner.

### The attempted fix: replay buffer + explicit rebalancing

Two changes were made together:

1. **A larger replay buffer** (8,000 examples spanning several recent iterations, not just
   the newest 20 games) - intended to give the rarer O-defense examples more chances to be
   reused across training steps instead of being diluted by whatever the latest small,
   possibly-skewed batch happened to contain.
2. **Explicit rebalancing by mover** (`balance_by_mover()`) - before each training step,
   oversample (with replacement) whichever side has fewer examples in the buffer, so every
   batch has an exactly equal split of X-to-move and O-to-move positions:

```python
def balance_by_mover(examples):
    by_mover = {X: [...], O: [...]}          # split the buffer by whose turn each example was
    majority_side = whichever has more examples
    minority_side = the other one
    oversampled_minority = random.choices(by_mover[minority_side], k=len(by_mover[majority_side]))
    return by_mover[majority_side] + oversampled_minority
```

Verified mechanically correct in a smoke test - batches came out perfectly balanced
(52/52, then 104/104, then 148/148 as the buffer grew) - and a full 300-iteration run was
launched with this fix in place.

### The result: partially helped, didn't fix the core problem

```
Value error vs. solver (a genuine improvement over the unbalanced run):
  Unbalanced run: 0.738 (early)  ->  0.544 (late)
  Balanced run:   comparable early stage -> ~0.35-0.42 by iteration ~140 (notably better)

Win/draw distribution (did NOT improve - if anything, looked just as skewed):
  Balanced run, iterations ~127-146: X wins 16-20, O wins 0-2, almost every iteration
```

### Why the fix only worked on one axis, and what that reveals

This is the most important lesson in this whole document: **rebalancing a training batch
only changes how much weight each side's *existing* examples get during a gradient step -
it cannot manufacture examples that don't exist yet.** Oversampling O's games means the
network sees the same mediocre defensive attempts *more often*, not *better* ones. If O
has never actually played a genuinely strong defensive line in any collected game, no
amount of reweighting the examples that *were* collected can teach the network a defense
it was never shown.

The real bottleneck sits one stage upstream, in the self-play generation itself
(`03_self_play_training_loop.md`): if the current network is bad at defending as O,
self-play (which is *played by that same network*) keeps generating lopsided games
regardless of how the resulting data is later reweighted. Rebalancing fixed how evenly the
network's *value calibration* improved (a real, measured win) but could not fix the
*quality of play* asymmetry, because that asymmetry is baked in before the rebalancing
step ever runs.

### What would actually be needed to fix the underlying problem

Not attempted in this project, but the natural next ideas, each targeting the *generation*
stage rather than the *training* stage:
- Dedicated self-play games where the network is deliberately forced to defend against a
  strong, fixed opponent (e.g. the perfect solver, or a strong MCTS) as O specifically,
  rather than only ever seeing O-side experience generated by playing against itself.
- A larger network and/or many more self-play games - AlphaZero's real implementations use
  orders of magnitude more games and much larger networks; it's plausible the imbalance
  would eventually self-correct given enough scale, even without a structural fix (this
  project's earlier discussion of AlphaZero's own real color-imbalance, in the
  conversation record, makes exactly this point: the actual system shows a milder version
  of the same effect, presumably softened by scale).
- Explicitly tracking and reporting O-specific and X-specific value error (splitting
  `value_error_vs_solver` by mover, the same technique already used to diagnose the
  TD(0) "drawing trap" in `../08_td0_online_learning.md`) to directly measure whether a
  future fix attempt is closing the *right* gap, not just improving an aggregate number
  that could mask a persistent asymmetry.

## A third attempt: teaching O with the perfect solver directly

The two ideas above (targeting the generation stage, and splitting value error by mover)
were both tried together as a direct follow-up.

### The design

During self-play generation, O's **actual move** was replaced with the perfect solver's
choice (`teacher_side=O` in `play_one_selfplay_game`), while X continued to learn entirely
through its own guided search. Crucially, the **policy training target** at O's positions
was left unchanged - still the network's own MCTS visit-count distribution, not a copy of
the solver's choice. Only which move actually got *played* was overridden, so that the
game's continuation always reflected genuinely correct defense, giving the value head
real outcomes to learn from along a correct defensive line, without directly forcing the
policy head to imitate the solver.

### The result: real progress on one axis, none on the other

A full 300-iteration run (first-20 vs. last-20 average, matching the comparison format
used for the other two runs):

```
Value error, split by mover:
  First 20: value_error_X = 0.679, value_error_O = 0.661   (gap: 0.018)
  Last  20: value_error_X = 0.600, value_error_O = 0.596   (gap: 0.004 - essentially closed)

Win/draw distribution (X vs. the solver-controlled O):
  First 20 avg: X wins 7.45, O(solver) wins 8.65, draws 3.90
  Last  20 avg: X wins 15.65, O(solver) wins 0.70, draws 3.65
```

Two real, measured wins: the X/O **value-error gap genuinely closed** (down to a 0.004
difference, from 0.018), and X's win rate against the solver's own optimal defense climbed
substantially, consistent with X gradually learning the actual forced-win technique
against the best possible opponent, not just against a weak mirror of itself.

**But this only tells us the network's *value judgment* of O-positions improved - it does
not, by itself, tell us whether the network's own *policy* can actually defend well
without the solver's help.** That required a separate, direct test.

### The decisive test: can the network play O on its own now?

Loaded the final teacher-trained network and had it control **both** sides *itself* (no
solver assistance at all) against the perfect solver, 20 games each way:

```
Network playing O (its own policy) vs. the perfect solver playing X:  0 wins, 20 losses, 0 draws
Network playing X vs. the perfect solver playing O:                  20 wins,  0 losses, 0 draws  (confirms X still strong)
```

**Zero improvement in actual playable defense**, despite the closed value-error gap.

### Why: calibration and competence are different problems

The mechanism this reveals is precise: overriding O's *played move* to be objectively
correct gave the value head real, correct outcomes to learn from (fixing calibration) -
but the *policy* head was never directly told "the solver's move was the right answer,"
only ever trained on the network's own (still self-generated, still weak) search
conclusions at that position. The game's trajectory improved; what the network was
actually taught to *do* at that decision point did not.

This is the single clearest lesson from this whole build: **a network can become
significantly better at judging a position's value while learning nothing new about how
to act on that judgment**, if the training signal for the two heads isn't equally
corrected. The natural next fix - tried next, below - is to make the **policy target**
itself imitate the solver's chosen move directly (e.g. a strong one-hot label), rather
than only correcting which continuations get explored.

## A fourth wave of attempts: more exploration, more compute, separate networks, gated freezing

Four more isolated experiments (one changed variable each vs. the unbalanced baseline),
each run for the full 300 iterations, each then put through the exact same decisive test
used above - load the final network(s), no solver help, no exploration randomness, both
sides play themselves against the perfect solver, 20 games each way.

- **exploration**: same as balanced, but a much longer temperature-sampling window
  (`temperature_moves=20`, up from 6) and stronger Dirichlet noise (`noise_frac=0.4`, up
  from 0.25) - more forced exploration diversity throughout self-play generation, not just
  the first few moves.
- **o_compute**: O gets 2x the MCTS simulations per move during self-play generation
  (200 vs. X's 100) - more "thinking time" as a cheap way to find better moves to learn
  from, without touching training or rebalancing at all.
- **twonet**: two completely separate networks (different seeds, zero shared weights),
  one trained only on X's positions, one only on O's - removes any possibility that a
  shared network is "compromising" on O's behalf.
- **catchup**: same two-network split as twonet, plus a gate - X's network skips its
  gradient update entirely whenever `value_error_O - value_error_X > 0.05`, intended to
  let O's network "catch up" before X pulls further ahead.

### First-20 vs. last-20 average, all four

```
                value_error_vs_solver        value_error gap (O-X)      wins_X / wins_O (per iter, avg)
                first20    last20            first20     last20         first20        last20
exploration     0.6965  -> 0.3930            +0.0149  -> -0.0018        9.75 / 8.70 -> 18.15 / 1.05
o_compute       0.6795  -> 0.4220            -0.0150  -> -0.0155        6.30 / 7.10 -> 18.95 / 0.45
twonet          0.7622  -> 0.4832            +0.1536  -> -0.0208        13.70 / 5.30 -> 18.50 / 0.85
catchup         0.7263  -> 0.4778            -0.0178  -> -0.1088        11.10 / 7.30 -> 18.65 / 0.50
```

Every single one of these shows the exact same shape already seen in `balanced`: value
error vs. the solver drops substantially (real learning is happening), and the X/O
calibration *gap* ends up small or even negative (O's calibration matching or beating
X's) - yet the win/draw distribution still collapses to X dominating almost every game by
the end. Calibration gap closing was never the bottleneck; it was a red herring the whole
time, confirmed four more times.

### catchup's freeze gate specifically: how often did it fire?

Out of 300 iterations, `x_frozen` was `True` for only **9 iterations total**, each a
brief, isolated blip that reverted a few iterations later - never a sustained freeze. The
reason is visible directly in the gap column above: `gap = value_error_O - value_error_X`
was **negative for essentially the entire run** (mildly at first, reaching as low as
-0.29 partway through) - meaning O's value calibration was, by this metric, already as
good as or better than X's the vast majority of the time. The gate is watching value
calibration, not win rate, and those two were never durably out of sync in the direction
the gate checks for - so it had almost nothing to react to, even while O kept losing
18-19 games out of 20 the entire second half of the run. This is not a bug in the gating
logic; it is the gate correctly answering a question ("is calibration lagging?") that
turns out to be the wrong question for the actual problem ("is O's policy competitive?").

### The decisive test: all four, network plays both sides itself, no help, vs. the perfect solver

```
                    Network as X (vs solver O)      Network as O (vs solver X)
exploration         20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
o_compute           20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
twonet              20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
catchup             20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
```

Identical to the teacher run's decisive-test result, and identical to each other: **every
single variant tried in this project, up through this wave, loses to the perfect solver
100% of the time when playing O on its own.** More exploration noise, more search compute
for O specifically, fully separate networks with zero shared weights, and gating X's
training off a calibration signal - none of these four independent mechanisms moved the
actual playable-defense number even slightly.

### The honest verdict across every approach tried so far

| Approach | Closed the value-error gap? | Fixed O's actual playable defense? |
|---|---|---|
| Balanced (rebalancing) | Improved aggregate error, gap not measured by mover | No |
| Teacher (solver plays O's moves) | Yes - gap closed to 0.004 | No - 0/20 |
| Exploration (longer noise window) | Yes - gap near zero | No - 0/20 |
| O-compute (2x search for O) | No change in gap direction | No - 0/20 |
| Twonet (separate networks) | Yes - gap closed and went negative | No - 0/20 |
| Catchup (gated freeze) | Yes - gap went strongly negative | No - 0/20 |

Six distinct, independently-tested mechanisms, all converging on the same conclusion:
**value calibration and playable policy competence are genuinely decoupled in this
build**, and nothing tried so far touches the policy head's actual training target at
O's positions. Every fix above operates on the value side (correcting outcomes, correcting
data balance, correcting *which* network sees what) - none of them ever tell the policy
head directly "this specific move was the solver's choice, reproduce it." That was the one
mechanism this project had identified but not yet tried - it was tried next, immediately
below, and it also failed.

## A fifth attempt: direct policy imitation - and a surprising negative result

The most promising untried idea, per the analysis above, was to stop only correcting
which move gets *played* at O's teacher-controlled positions and also directly correct
what the **policy head is trained to predict** there.

### The design

Same setup as the original `teacher` run (`teacher_side=O`, perfect solver plays O's
actual moves during self-play generation), with one addition: at every position where the
teacher plays, the recorded policy training target is overridden to a one-hot vector on
the solver's chosen move, instead of the network's own (self-generated, potentially still
weak) MCTS visit-count distribution:

```python
if is_teacher_move:
    move = perfect_solver_move(game)
    if imitate_teacher_policy:
        policy_target = np.zeros(POLICY_SIZE)
        policy_target[encode_action(move)] = 1.0   # directly: "this exact move was correct here"
```

This is a direct, mechanical fix for exactly the gap the teacher experiment identified:
the policy head now receives an explicit, unambiguous training signal at every teacher
position, not just a corrected game continuation for the value head to learn from.

### The result: value calibration behaved as expected, playable defense still did not improve

```
First-20 vs. last-20 average (300 iterations):
  value_error_vs_solver:  0.6810 -> 0.5396
  value_error_X:          0.6894 -> 0.5636
  value_error_O:          0.6902 -> 0.5314
  gap (O - X):            0.0007 -> -0.0322   (essentially closed, same pattern as plain teacher)

Win/draw distribution (X vs. solver-controlled O, during self-play generation):
  First 20 avg: X wins 7.95, O(solver) wins 6.95, draws 5.10
  Last  20 avg: X wins 16.20, O(solver) wins 2.05, draws 1.75
```

The decisive test - network controls **both** sides itself, no solver help, vs. the
perfect solver, 20 games each way:

```
Network playing X vs. perfect solver playing O:  20 wins,  0 losses, 0 draws
Network playing O vs. perfect solver playing X:   0 wins, 20 losses, 0 draws
```

**Still zero improvement in actual playable defense.** Directly one-hotting the policy
target on the solver's move - the fix specifically designed to close the gap this whole
document had been building toward - produced exactly the same 0/20 result as every prior
attempt.

### Why this is a genuinely surprising result, and what it implies

This is not what the calibration-vs-competence framing predicted. If the policy head
really just needed the correct label at teacher-visited positions, this fix should have
worked, or at least partially worked, more than the others. It didn't move the needle at
all. Plausible explanations, none yet tested against each other:

1. **Coverage, not signal quality, is the bottleneck.** A one-hot label at the specific
   positions the teacher happens to visit during 300 iterations x 20 games is still only
   ever training on whatever narrow slice of O's total decision space self-play happens to
   reach - and self-play reaches a narrow slice precisely *because* X keeps winning
   quickly. Correcting the label at those few positions doesn't help the network generalize
   to positions it was never shown at all, and a 64-hidden-unit network with only ~4,974
   total states in the whole game may still not have enough varied exposure to O's
   correct defensive lines to generalize the *pattern* behind them, only memorize the
   specific instances seen.
2. **The bottleneck may be upstream of the policy target entirely, in the search/MCTS
   step during self-play itself, not the training step.** Even with a corrected policy
   target, if the network's initial priors and MCTS's exploration during that same
   self-play generation never surface the objectively strong response as a real candidate
   often enough (games still end quickly because O's non-teacher moves in other games -
   any position not exactly matching a teacher-controlled turn - are still guided by the
   network's own weak understanding), the total volume of well-labeled training data may
   simply be too small relative to the size of the problem.
3. **Network capacity.** This project deliberately used a small network (64 hidden units,
   two hidden layers) to keep iteration times short for rapid experimentation
   (`05_results_summary.md`). It's possible this capacity is sufficient for the *value*
   function (a single scalar, well fit already - Stage 1 achieved 0.206 test MAE) but
   insufficient for the *policy* function to represent the more intricate move-selection
   logic real defense requires, especially split across many distinct board
   configurations.
4. **The asymmetry may simply require far more self-play games than this project's
   budget** (6,000 total games per run) - AlphaZero's real implementations use many orders
   of magnitude more self-play games and much larger networks; the imbalance measured here
   might be a small-scale-and-short-budget artifact that would eventually correct itself
   given enough scale, independent of which of the seven training-signal fixes is used.

None of these four have been tested against each other yet - doing so (e.g. a much larger
replay of self-play games focused specifically on O's positions, or a deliberately larger
network) is the natural next step, but is a materially bigger experiment than anything
tried in this document so far.

### Updated final tally: seven approaches, seven failures on the decisive test

| Approach | Closed the value-error gap? | Fixed O's actual playable defense? |
|---|---|---|
| Balanced (rebalancing) | Improved aggregate error, gap not measured by mover | No |
| Teacher (solver plays O's moves) | Yes - gap closed to 0.004 | No - 0/20 |
| Exploration (longer noise window) | Yes - gap near zero | No - 0/20 |
| O-compute (2x search for O) | No change in gap direction | No - 0/20 |
| Twonet (separate networks) | Yes - gap closed and went negative | No - 0/20 |
| Catchup (value-calibration-gated freeze) | Yes - gap went strongly negative | No - 0/20 |
| **Imitate (direct policy one-hot on teacher's move)** | Yes - gap closed to -0.032 | **No - 0/20** |

Seven distinct, independently-tested mechanisms, all landing on the exact same decisive
result. This project has now ruled out training-data rebalancing, solver-corrected
continuations, more exploration noise, more search compute for the weaker side, fully
separate networks, two different freeze-gating strategies (value-calibration-based and,
separately, win-rate-based - see `05_results_summary.md`), and direct policy-target
imitation. The honest conclusion at this point is that **the remaining bottleneck is most
likely scale (self-play volume and/or network capacity), not any single training-signal
mechanic** - every mechanic tried operates on the same modest data budget and small
network, and none of them broke through it.

## What's next

`05_results_summary.md` collects every concrete number produced across this whole build,
compared against each other and against the earlier table-based approaches in this
project.
