# Results: Every Number This Build Produced

Read `00` through `04` first for the full explanation behind each of these results. This
document is the concrete evidence, collected in one place.

## Stage 1: does the network actually generalize?

Trained the value head alone on 80% of the 4,974 exactly-solved positions
(`../04_perfect_solver_exact_solution.md`), tested on the 20% held out and never shown
during training:

| Epoch | Test MAE (unseen states) |
|---|---|
| 1 | 0.566 (essentially random guessing) |
| 10 | 0.299 |
| 20 | 0.246 |
| 30 | 0.226 |
| 40 | 0.214 |
| 50 | 0.210 |
| 60 | **0.206** |

**Verdict**: real, substantial generalization to positions never seen during training -
something the table-based Q-learning approach used throughout the rest of this project
can never do by construction (an unvisited state has literally zero information in a
table; here it gets a genuinely useful estimate).

## Stage 2: guided MCTS, before and after the terminal-value bug fix

20 games against a fully random opponent, value-only-trained network, 100 MCTS
simulations per move:

| | Wins | Losses | Draws | Time for 20 games |
|---|---|---|---|---|
| **Before fix** (sign bug present) | 0 | 14 | 6 | 11.8s |
| **After fix** | 20 | 0 | 0 | 2.4s |

Also markedly faster than the project's earlier rollout-based plain MCTS (`mcts.py`) at
comparable iteration counts, since a single network call replaces simulating an entire
game to its conclusion.

Controlled one-move-win test case (see `04_bugs_lessons_and_limitations.md` for the full
board), confirming the fix specifically:

| | Visits on the winning move (move 2) | Move actually chosen |
|---|---|---|
| Before fix | 1 (found, then abandoned) | move 6 (not a win) |
| After fix | 193 / 200 | move 2 (correct) |

## Stage 3: the full self-play loop, three variants compared

All three runs: 300 iterations, 20 self-play games per iteration (6,000 total games), 100
MCTS simulations per move, identical network architecture and random seed for
initialization.

### Value error vs. the exact solver

| | First 20 iterations (avg) | Last 20 iterations (avg) |
|---|---|---|
| **Unbalanced** (fresh examples only, no rebalancing) | 0.738 | 0.544 |
| **Balanced** (8,000-example replay buffer + X/O rebalancing) | 0.691 | 0.399 |
| **Teacher-assisted** (solver plays O's moves during generation) | 0.671 | **0.594** (aggregate) |

The teacher-assisted run's *aggregate* value error looks worse than the balanced run's -
but this is misleading in isolation; see the by-mover split below, which is the number
that actually matters for this question.

### Value error, split by mover (X-to-move vs. O-to-move) - teacher-assisted run only

| | First 20 iterations (avg) | Last 20 iterations (avg) |
|---|---|---|
| value_error_X | 0.679 | 0.600 |
| value_error_O | 0.661 | **0.596** |
| **gap (X - O)** | 0.018 | **0.004** |

The gap between X's and O's calibration accuracy genuinely closed to almost nothing - a
real, measured success on this specific axis, and the first run in this whole project to
achieve it.

### Win / draw distribution

| | First 20 iterations (avg per iteration) | Last 20 iterations (avg per iteration) |
|---|---|---|
| **Unbalanced**: X wins | 9.10 | 17.20 |
| **Unbalanced**: O wins | 7.35 | 1.40 |
| **Unbalanced**: draws | 3.55 | 1.40 |
| **Balanced**: X wins | 10.15 | 18.65 |
| **Balanced**: O wins | 5.25 | **0.15** |
| **Balanced**: draws | 4.60 | 1.20 |
| **Teacher-assisted**: X wins (vs. solver-controlled O) | 7.45 | 15.65 |
| **Teacher-assisted**: O (solver) wins | 8.65 | 0.70 |
| **Teacher-assisted**: draws | 3.90 | 3.65 |

The simple rebalancing fix (**Balanced**) did not resolve the imbalance - if anything, O's
win rate ended up lower than the unbalanced run. The **teacher-assisted** run shows X
steadily learning to beat the solver's own genuinely optimal defense (win rate roughly
doubling), which is a meaningfully different and more informative signal than beating a
weak, self-generated O.

### Confirming the (balanced run's) imbalance is real, not exploration noise

Both sides forced to play fully deterministically (no temperature sampling, no Dirichlet
noise) with the final balanced-run network, 20 games:

```
Result: X wins 20/20, O wins 0/20, draws 0/20
```

### The decisive test: did the teacher-assisted run actually fix playable defense?

Closing the value-error gap is necessary but not sufficient - the real question is whether
the network can defend *on its own*. Loaded the teacher-assisted run's final network and
had it control **both** sides itself (no solver assistance), 20 games each way, against the
perfect solver:

```
Network playing O (its own policy, no help) vs. perfect solver playing X:  0 wins, 20 losses, 0 draws
Network playing X vs. perfect solver playing O:                           20 wins,  0 losses, 0 draws
```

**No improvement in actual playable defense, despite the closed value-error gap.** The fix
improved the network's *judgment* of O-positions (the value head) without improving its
*behavior* at those positions (the policy head), because only the game's continuation was
corrected during training, not the direct policy training target. See
`04_bugs_lessons_and_limitations.md` for the full mechanism and the proposed next fix
(imitating the solver's chosen move directly in the policy target, not just the
continuation).

### The full 2x2 picture: does this generalize beyond self-play, to any opponent?

A natural follow-up: is this weakness specific to playing against itself, or would it show
up against a real, independent opponent too - i.e. would a human notice it? Tested the
final balanced-run network against two very different opponents it had never played
during training, on both sides:

```
                        vs random (weak)            vs perfect solver (strongest possible)
Network playing X:      19/20 win, 1 draw           20/20 win  <- beats even perfect play
Network playing O:       20/20 win                   0/20 win, 20 losses
```

This is a clean, complete answer: the network's X-side play has genuinely mastered the
real forced win proven in `../04_perfect_solver_exact_solution.md` - it wins even against
a mathematically perfect defender. Its O-side play beats a careless/random opponent
comfortably, but has learned no defensive technique that survives contact with real
skill - it loses to literally every opponent tried except random play. Critically, this
means the weakness is **not an artifact of self-play specifically** - it is a genuine gap
in the network's learned knowledge, and would show up identically against a human playing
well, exactly as it does against the perfect solver here.

Removing all randomness made the imbalance *more* extreme, not less - conclusively ruling
out "just exploration noise" as the explanation.

## Stage 4: four more attempts at the asymmetry (exploration, o_compute, twonet, catchup)

Four further, isolated experiments (one changed variable each, per the project's rule of
never mixing hypotheses in a single run), all 300 iterations, same base architecture:

- **exploration**: same as balanced, but a much longer temperature-sampling window
  (`temperature_moves=20`, up from 6) and stronger Dirichlet noise (`noise_frac=0.4`, up
  from 0.25).
- **o_compute**: identical to the unbalanced run, except O gets 2x the MCTS simulations
  per move during self-play generation (200 vs. X's 100) - the idea being that if O is
  structurally at a disadvantage, giving it more search compute might let it find better
  moves to learn from.
- **twonet**: two completely separate `TicTacToeNet` instances (different random seeds,
  no shared weights at all) - one trained only on X's positions, one only on O's. Removes
  any possibility that a shared network is "compromising" on O's behalf to also serve X.
- **catchup**: same two-network split as twonet, but X's network training is frozen
  (skips its gradient update that iteration) whenever `gap = value_error_O - value_error_X`
  exceeds 0.05 - the idea being to let O's network "catch up" before X pulls further ahead.

### First-20 vs. last-20 iteration averages, all four

```
                value_error_vs_solver        value_error gap (O-X)      wins_X / wins_O (avg per iter)
                first20    last20            first20     last20         first20        last20
exploration     0.6965  -> 0.3930            +0.0149  -> -0.0018        9.75 / 8.70 -> 18.15 / 1.05
o_compute       0.6795  -> 0.4220            -0.0150  -> -0.0155        6.30 / 7.10 -> 18.95 / 0.45
twonet          0.7622  -> 0.4832            +0.1536  -> -0.0208        13.70 / 5.30 -> 18.50 / 0.85
catchup         0.7263  -> 0.4778            -0.0178  -> -0.1088        11.10 / 7.30 -> 18.65 / 0.50
```

Every one of these four shows real learning (value error drops substantially) and the
X/O calibration gap ending up small or negative (O calibrated as well as or better than
X) - yet the self-play win distribution still collapses to X dominating by the end in
every case. The gap metric alone would suggest all four "worked" - the decisive test
below shows they didn't.

### The decisive test, all four: network plays both sides itself, no help, vs. the perfect solver

```
                    Network as X (vs solver O)      Network as O (vs solver X)
exploration         20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
o_compute           20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
twonet              20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
catchup             20 wins,  0 losses, 0 draws     0 wins, 20 losses, 0 draws
```

Identical to the teacher run's decisive-test result (Stage 3 above), and identical to
each other. Longer exploration, more search compute for O, fully independent networks,
and gating X's training on a calibration signal - **none of the four moved the
playable-defense number even slightly.**

### Why catchup's freeze mechanism never engaged

Across all 300 iterations, `x_frozen` was `True` for only **9 iterations total**, each a
brief, isolated blip that reverted a few iterations later. For the large majority of the
run - especially the entire back half, iterations ~100-300 - `gap` was **negative**
(commonly -0.05 to -0.29, e.g. iteration 269: `gap=-0.2942`), meaning O's value
calibration was *already as good as or better than* X's by the metric the gate watches.
The freeze condition (`gap > 0.05`) is watching **value calibration**
(`value_error_vs_solver_by_mover`), not **win rate**. This is the same calibration-vs-
competence split first found in the teacher-assisted experiment (Stage 3): a network can
correctly judge that a position is losing while still having no idea which move
minimizes the damage. Since O's calibration was never durably behind X's, the gate
correctly concluded "no freeze needed" - while O kept losing 18-19 games out of 20 the
entire time. The mechanism isn't buggy; it's gating on the wrong signal for the problem
it was built to solve.

## Stage 5: direct policy imitation (imitate) - and a second gating strategy (catchup_wr)

Two more targeted attempts, each addressing a specific weakness identified above.

### imitate: one-hotting the policy target on the teacher's actual move

Same setup as `teacher`, but at every position where the solver plays O's move, the
**policy training target** is also overridden to a one-hot vector on that move - directly
telling the policy head "this was correct here," not just correcting the game's
continuation for the value head.

```
First-20 vs. last-20 average (300 iterations):
  value_error_vs_solver:  0.6810 -> 0.5396
  value_error_X:          0.6894 -> 0.5636
  value_error_O:          0.6902 -> 0.5314
  gap (O - X):            0.0007 -> -0.0322

Decisive test (network plays both sides, no help, vs. perfect solver):
  Network as X:  20 wins,  0 losses, 0 draws
  Network as O:   0 wins, 20 losses, 0 draws
```

**Zero improvement**, despite this being the single most targeted fix attempted - it
directly patches the exact gap (policy target never corrected) that the teacher
experiment's decisive test had identified as the missing piece. See
`04_bugs_lessons_and_limitations.md` for the full discussion of why this is a genuinely
surprising negative result and what it implies about the real bottleneck (likely scale -
self-play volume and/or network capacity - rather than any single training-signal fix).

### catchup_wr: gating on win rate instead of value calibration

The plain `catchup` run's freeze gate almost never engaged because it watched
`value_error_O - value_error_X`, a metric that turned out to be decoupled from win rate.
`catchup_wr` uses the same two-separate-networks setup, but gates X's training directly
on O's own rolling win rate (10-iteration window), with hysteresis: freezes below a 15%
win rate, only unfreezes above 30% (asymmetric thresholds specifically to prevent rapid
flapping in between). The freeze mechanism was unit-tested against synthetic win-rate
sequences and smoke-tested against real (small-scale) training - confirmed to actually
skip X's gradient updates when frozen, not just log the flag - before launching the full
run. Results pending (run in progress as of this writing).

### The honest verdict, seven approaches compared

| Approach | Closed the value-error gap? | Fixed O's actual playable defense (decisive test)? |
|---|---|---|
| Balanced (rebalancing) | Improved aggregate error | No |
| Teacher (solver plays O's moves) | Yes - gap closed to 0.004 | No - 0/20 |
| Exploration (longer noise window) | Yes - gap near zero | No - 0/20 |
| O-compute (2x search for O) | No change in gap direction | No - 0/20 |
| Twonet (separate networks) | Yes - gap closed, went negative | No - 0/20 |
| Catchup (value-calibration-gated freeze) | Yes - gap went strongly negative | No - 0/20 |
| Imitate (direct policy one-hot on teacher's move) | Yes - gap closed to -0.032 | No - 0/20 |

Seven independently-tested mechanisms, one conclusion: value calibration and playable
policy competence are decoupled in this build, and - more importantly, after the
`imitate` result - even directly fixing the policy label at every teacher-visited
position doesn't help either. The bottleneck most likely isn't which training signal is
used at all, but the fundamental scale this project runs at (6,000 self-play games and a
64-hidden-unit network per run, both orders of magnitude below a real AlphaZero-style
system). See `04_bugs_lessons_and_limitations.md` for the full reasoning.

## Stage 6: broad supervised policy imitation (stage1_policy_imitation.py) - an 8th attempt

Every prior policy-imitation attempt (`teacher`, `imitate`) only ever showed the policy
head a correct answer at whatever handful of positions self-play happened to visit -
sparse, incidental coverage gated behind the self-play process itself. Stage 1 (top of
this document) proved broad, direct supervision on thousands of solved positions - no
self-play involved at all - genuinely generalizes for the *value* head. This experiment
asks the natural next question: does the same recipe generalize for the *policy* head too?

### The design

Sampled all 4,974 solved states directly (no self-play), split 80/20 train/held-out. For
each state, computed every move tied for the maximum value (ties matter here - punishing
the network for confidently choosing a different, equally-optimal move would be an
unfair target) via a fixed `all_optimal_moves()` helper, and trained both heads together:
the policy head against a uniform distribution over the optimal-move set, the value head
against the solver's exact value - pure supervised learning, 60 epochs, no MCTS, no
self-play loop.

### The result: real generalization, on the policy head, for the first time in this project

```
epoch  1/60: held-out value_MAE=0.5903  policy_top1_acc=0.6503
epoch 20/60: held-out value_MAE=0.3456  policy_top1_acc=0.8784
epoch 60/60: held-out value_MAE=0.2506  policy_top1_acc=0.9286
```

**92.9% policy accuracy on states never seen during training** - a genuine result, and
categorically different from every self-play-based imitation attempt: this is the first
time in the whole project a policy head demonstrably generalized to unseen positions,
rather than only ever being corrected at positions self-play happened to reach.

### The decisive test: still 0/20 as O

```
Network as X vs perfect solver as O: 20 wins,  0 losses, 0 draws
Network as O vs perfect solver as X:  0 wins, 20 losses, 0 draws
```

Despite the strong, validated held-out accuracy, and playing with full MCTS search (100
iterations) rather than the raw policy alone, the network still loses every single game
as O.

### Why: 93% per-move accuracy is nowhere near enough against a perfect adversary

This is the real, previously-missing piece of the picture. A full game is a *sequence* of
moves, and the perfect solver isn't a random source of errors to get lucky against - it
actively steers the game toward whichever position is most likely to expose whatever
error rate remains. O's task is to play an **entire game with zero mistakes** to force a
draw (the sliding game's proven ceiling for O, `04_perfect_solver_exact_solution.md`); a
~7% per-move error rate, hunted for specifically rather than encountered by chance, is
enough to lose essentially every time, even though 93% looks like a strong number in
isolation. Naive independent-error-rate arithmetic (0.93^7 ≈ 60% for a 7-move game) even
*understates* the real difficulty, since an adversarial opponent doesn't sample errors
uniformly at random - it specifically searches for and routes into the network's weakest
positions.

### The honest reframing this produces

This changes the shape of the whole X/O asymmetry question. It was never really about
*whether* a network can learn the correct move most of the time - broad supervision
answers that "yes," decisively, for the first time in this project. The real bottleneck
is that "mostly correct" and "perfect for an entire game against an adversary actively
hunting for the one mistake" are two different bars, and only the second one actually
beats a perfect opponent. Every fix attempted in this project (Stages 3-6, eight
approaches total) improved calibration, or generalization, or accuracy - useful, real
progress on each of those - without ever closing the gap to the much harder bar required.

### Updated final tally: eight approaches, eight failures on the decisive test

| Approach | What it improved | Fixed O's actual playable defense (decisive test)? |
|---|---|---|
| Balanced (rebalancing) | Aggregate value error | No |
| Teacher (solver plays O's moves) | Value-error gap (closed to 0.004) | No - 0/20 |
| Exploration (longer noise window) | Value-error gap (near zero) | No - 0/20 |
| O-compute (2x search for O) | Nothing measurable | No - 0/20 |
| Twonet (separate networks) | Value-error gap (closed, went negative) | No - 0/20 |
| Catchup (value-calibration-gated freeze) | Nothing (gate rarely triggered) | No - 0/20 |
| Imitate (self-play-gated policy one-hot) | Value-error gap (closed to -0.032) | No - 0/20 |
| **Broad supervised policy imitation** | **Held-out policy accuracy (92.9%)** | **No - 0/20** |

## Comparison to the rest of this project's approaches

| Approach | Generalizes to unseen states? | Needs an opponent to train against? | Needs the solver during training? | Known failure mode found |
|---|---|---|---|---|
| Sparse Q-learning (table) | No | Yes (fixed) | No | None found |
| Shaped Q-learning (table, solver-based) | No | Yes (fixed) | Yes | Collapses vs. an imperfect opponent (`../06`) |
| TD(0) (table, online) | No | Yes (fixed) | No | "Drawing trap" - rare wins never get learned (`../08`) |
| Self-play TD(0) (table) | No | No (plays itself) | No | Not yet fully diagnosed (see `../08`) |
| **This build (network + guided MCTS + self-play)** | **Yes** | No (plays itself) | Only for *measuring* progress, not training | X/O skill asymmetry - 8 distinct fix attempts (rebalancing, teacher, more exploration, more O compute, separate networks, freeze-and-catchup gating, self-play-gated policy imitation, broad supervised policy imitation with 92.9% held-out accuracy), all verified via the decisive test (network plays both sides, no help, vs. solver) to still lose 0/20 as O - the real bottleneck turned out to be that "mostly correct" per move is nowhere near "perfect for an entire game vs. an adversary hunting for the one mistake" |

Every approach tried in this project has hit a real, measured limitation - which is itself
the throughline of this whole body of work: theory alone (a proof that a technique is
"correct," like potential-based shaping's policy-invariance guarantee, or the intuitive
appeal of "just add a neural network") is never sufficient by itself. Every single
technique here needed to be run, measured against the exact solver or a controlled test,
and often debugged, before its real behavior could be trusted.

## What's still open

1. **RESOLVED: splitting value error by mover** - applied directly (`value_error_vs_solver_by_mover`),
   and it revealed something the aggregate number couldn't: the teacher-assisted run
   closed the X/O *calibration* gap almost completely (0.004 difference by the end), a real
   success invisible in the plain aggregate metric alone.
2. **NEW, the actual remaining problem: policy imitation, not just value calibration.**
   The teacher-assisted run proved that correcting a position's *outcome* (via a solver
   dictating O's actual moves) improves the value head without teaching the policy head to
   reproduce that behavior. The natural next fix - not yet attempted - is to make the
   policy training target at the teacher-controlled side's positions directly imitate the
   solver's chosen move (e.g. a strong one-hot label, possibly blended with the search's
   own visit distribution), rather than leaving that target entirely self-generated.
3. **Does the X/O playable-defense gap resolve with more scale** (more self-play games, a
   larger network) even without a direct policy-imitation fix? Not yet tested - all seven
   variants ran the same 300-iteration budget.
3b. **RESOLVED (negative result): more O-side search compute, fully separate networks,
   and freeze-and-catchup gating do not fix it either.** Three further isolated
   experiments (Stage 4 above) each changed exactly one variable relative to the
   unbalanced baseline and each still ended at roughly X=19-20, O=0-1 out of 20. The
   catchup run additionally showed that gating on value-calibration gap doesn't track
   win-rate gap at all - the two diverge, confirming (a second time, independently of the
   teacher experiment) that this project's core remaining problem is specifically
   *policy* imitation/competence, not value calibration, and that none of the six fixes
   attempted so far target the policy head directly. The one fix outlined in point 2
   above (imitating the solver's actual move choice in the policy training target)
   remains the only untried approach that addresses the right layer of the network.
4. **A larger network and/or MCTS iteration count** - this build deliberately used a small
   network (64 hidden units) and modest search budget (100 simulations/move) to keep
   iteration times short enough for rapid experimentation; whether a larger version closes
   the gap to the Stage 1 ceiling (0.206) or the asymmetry gap is unmeasured.
5. **Eligibility traces / TD(lambda)**, discussed conceptually earlier in this project's
   conversation record as a way to make rare wins propagate credit further per occurrence,
   was never actually implemented for this network-based approach - only discussed for the
   table-based TD(0) agent.
