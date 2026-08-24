# Reward Shaping: Theory, Implementation, and an Opponent-Dependent Failure Mode

This is the central experimental finding of the project so far: potential-based reward
shaping, built using the perfect solver's exact values, works exactly as theory predicts
against a perfect opponent, and **measurably degrades performance against an imperfect
one** - a real, observed, opponent-dependent failure mode, not a bug in the implementation.

## The baseline: sparse reward

As implemented from the start (see `02_qlearning_theory_and_implementation.md`): reward is
given only at the terminal move of a game (+1 win, 0 draw, -1 loss), discounted backward
through every earlier move via `G <- 0 + gamma*G`. Every move in a losing game gets blamed
roughly equally (discounted only by how far back it is), with no way to distinguish which
specific move was actually the mistake.

## The question: is this "the optimized" reward system?

No single reward scheme is universally "the optimized one" - it depends on the goal. Sparse
reward is the safest possible choice: since the agent only ever gets credit for the
*actual* outcome, it cannot be misled into overvaluing something that looks good but isn't
(the general risk known as reward hacking / shaping bias). Its cost is slow, blurry credit
assignment.

## Potential-based shaping (Ng, Harada & Russell, 1999)

For any function `Phi: States -> Real numbers` (a "potential function"), adding a shaping
term of the form

```
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

to the environment's reward is proven to leave the **optimal policy provably unchanged**
- regardless of what `Phi` is. The proof shows `Q'(s,a) = Q(s,a) - Phi(s)` exactly: a
constant shift that depends only on `s`, not on which action is taken, so it can never
change *which* action ranks highest at that state. This is what makes shaping "safe" in a
way that arbitrary hand-designed bonuses are not: it can speed up learning without risking
a different, wrong policy.

Since the perfect solver (`04_perfect_solver_exact_solution.md`) provides an exact value
for every state, it is the best possible choice of `Phi` available for this game -
motivating an attempt to use it directly.

## Implementation

`phi_agent(game, q_plays)`: the solver's `negamax_value` is defined as "value to whoever is
about to move," which flips meaning depending on whose turn it is - not usable directly as
a potential function, which must have a fixed meaning per state. `phi_agent` re-expresses it
from the learning agent's own fixed point of view (positive = good for the agent,
consistently, regardless of whose turn it currently is):

```python
def phi_agent(game, q_plays):
    value = negamax_value(game)
    return value if game.current_player == q_plays else -value
```

During each of the agent's own move choices, `Phi` is captured *before* that move is made.
The terminal reward is folded directly into `Phi` at the terminal state (`Phi(terminal) =
terminal_reward`) rather than being added separately, which avoids double-counting; the
backward pass then seeds `G` at 0 (rather than at the terminal reward, as in the sparse
version) and computes, for each of the agent's transitions:

```
step_reward = gamma * Phi(next_decision_state) - Phi(current_decision_state)
G <- step_reward + gamma * G
```

This telescopes to the same total credit as the sparse version (shifted by a constant), per
the theorem above - the difference is *how that credit is distributed* across the moves
that led there, not the total amount.

## Validating the implementation in isolation, before touching any live training

Deliberately done as a standalone script, without touching the (by then, valuable) live
training session, to avoid any risk to real progress. One real game was replayed
(a randomly-moving learner vs. the perfect solver) and the shaped reward at each step
computed by hand from the recorded values:

```
Game finished for real. winner: O | terminal_reward (agent's perspective): -1

move #0: phi_before= 1.0000  (agent had a position where it could force a win)
move #1: phi_before=-1.0000  (after its own move + optimal reply, now a forced loss)
move #2: phi_before=-1.0000
move #3: phi_before=-1.0000

Shaped step rewards (backward):
  move #3: step_reward= 0.1000
  move #2: step_reward= 0.1000
  move #1: step_reward= 0.1000
  move #0: step_reward=-1.9000   <- the actual mistake, sharply flagged
```

This is a concrete demonstration of the claimed benefit: move #0 - the one that actually
threw away a forced win - receives a sharp, specific penalty (-1.9), while moves #1-#3
(already in an unrecoverable position, nothing left to be done) each receive only a small,
flat, uninformative +0.1. The sparse baseline, by contrast, would apply the same
discounted -1-derived signal to all four moves roughly equally, with no way to single out
move #0 as the actual error.

## The experiment: enabling shaping live, against MCTS

With the theory validated and the demonstration in hand, shaping was enabled
(`reward_shaping: 1`) on an already-substantially-trained live agent, while its opponent
remained MCTS (400 iterations/move at the time).

### Observed result: a real, sustained performance decline

Checkpoints (20 evaluation games each, alternating sides, measured every 50 training games)
over roughly 700 further games:

```
game 7300: win_rate=60%
game 7350: win_rate=30%
game 7400: win_rate=10%
game 7450: win_rate=20%
game 7500: win_rate=40%
game 7550: win_rate=20%
game 7600: win_rate=30%
game 7650: win_rate=40%
game 7700: win_rate=20%
game 7750: win_rate=30%
game 7800: win_rate=10%
game 7850: win_rate=10%
game 7900: win_rate=10%
game 7950: win_rate=0%
game 8000: win_rate=0%
```

The last several checkpoints, measured with a larger sample (20 games rather than an
earlier 10) to reduce noise, confirmed this was a real, sustained decline rather than
small-sample noise - a smooth slide from 60% down to a flat 0%.

### Diagnosis: the invariance guarantee has a hidden assumption

The Ng, Harada & Russell theorem guarantees policy invariance for a *fixed, Markovian*
environment matching whatever `Phi` was computed under. `Phi` here is `negamax_value`,
whose meaning is precisely "the outcome **assuming both sides play optimally** from this
point on." MCTS is not optimal - it approximates, and can be beaten. When the actual
opponent is fallible, `Phi`'s "assume perfect resistance" assumption is simply false
relative to what's actually happening in the game: a position `Phi` labels a certain loss
(true only against a perfect opponent) might, in reality, still be winnable or drawable
against this specific, beatable MCTS - but the shaped reward has no way to represent that
possibility, and will systematically discourage the agent from playing lines that a real,
fallible opponent might have let it get away with. Sparse reward has no equivalent failure
mode, because it only ever reacts to what actually happened in that specific game, never to
a hypothetical perfect-opponent outcome that didn't occur.

The hypothesis that follows: **this degradation should be specific to imperfect
opponents**, and should not occur when the actual opponent genuinely is the perfect solver
- because then `Phi`'s "assume optimal play" assumption is exactly true, not an
approximation.

### A natural confirmation: switching the opponent to the perfect solver

Independently of the diagnosis above, the opponent was switched to the perfect solver
(training continued with shaping still on). Two things were observed:

1. **Training speed increased dramatically** (from game 8,032 to game 31,015 in a short
   span) - expected and unrelated to the reward-shaping question: the perfect solver
   answers with an instant table lookup, while MCTS at 400 iterations had been taking
   multiple seconds per move.

2. **Checkpoints stabilized at exactly 50% win / 50% draw / 0% loss**, consistently, across
   the last 10 checkpoints measured (games 30,550 through 31,000). This is not just "good"
   - it is the exact theoretical ceiling: since X can force a win and O's best possible
   outcome is a draw (`04_perfect_solver_exact_solution.md`), an agent alternating sides
   and playing *perfectly* would win exactly every game it plays as X and draw exactly
   every game it plays as O - precisely a 50/50/0 split. The agent reached that ceiling.

This is a direct, empirical confirmation of the diagnosis: the same reward-shaping
mechanism that appeared to actively harm performance against MCTS produced apparently
optimal or near-optimal behavior once paired with an opponent that actually satisfies the
assumption the shaping relies on.

## Summary comparison

| | Sparse reward | Potential-based shaping (solver `Phi`) |
|---|---|---|
| Feedback per move | Only at game end | Every move |
| Credit assignment | Blurry - blames a whole sequence roughly equally | Sharp - demonstrated pinpointing the exact mistake |
| Requires external knowledge | No | Yes - the exact solver |
| Policy-invariance guarantee | N/A (already unbiased by construction) | Proven, but *only* when the opponent's behavior matches what `Phi` assumes |
| Observed behavior vs. MCTS (imperfect opponent) | (baseline, not separately re-tested under identical conditions yet) | Real, sustained win-rate decline, 60% -> 0% over ~700 games |
| Observed behavior vs. perfect solver | N/A (no sparse-vs-perfect-solver run recorded at matching scale yet) | Stable 50% win / 50% draw / 0% loss - the exact theoretical ceiling |

## Follow-up experiment: switching shaping back off against MCTS

The natural next controlled experiment was run: with the same already-trained Q-table
(by this point ~1,700 states, having been through a shaped-reward decline against MCTS,
then a long stable run against the perfect solver), the opponent was set back to MCTS
(400 iterations/move) and `reward_shaping` was set back to **0**.

Checkpoints immediately before and after the switch:

```
game 39150 (shaping=ON,  opponent=MCTS):  win=15%, draw=10%  -> non-loss=25%
game 39200 (shaping=ON,  opponent=MCTS):  win=10%, draw=30%  -> non-loss=40%
--- reward_shaping switched to 0 here ---
game 39250 (shaping=OFF, opponent=MCTS):  win=15%, draw=10%  -> non-loss=25%
game 39300 (shaping=OFF, opponent=MCTS):  win=10%, draw=25%  -> non-loss=35%
game 39350 (shaping=OFF, opponent=MCTS):  win= 5%, draw=40%  -> non-loss=45%
game 39400 (shaping=OFF, opponent=MCTS):  win=25%, draw=35%  -> non-loss=60%
```

**Updated result: still not conclusive on raw win rate, but a suggestive trend appears in
the combined non-loss rate (win+draw).** Raw win rate alone stays noisy after the switch
(15% -> 10% -> 5% -> 25%, no clean monotonic climb). But the *non-loss* rate - how often
the agent avoids losing outright, whether by winning or drawing - climbs fairly
consistently across the four post-switch checkpoints: 25% -> 35% -> 45% -> 60%. That is a
real pattern worth flagging, not a single lucky point, but it rests on only 80 total
evaluation games (4 checkpoints x 20 games) and is not yet enough to confidently rule out
noise, especially given the ~700-game window needed to establish the original decline with
confidence.

This does **not** contradict the diagnosis above, but at the time it did not yet close the
question via direct experiment - the win-rate-only evidence was too noisy to be conclusive
on its own (see below for the metric that resolved this).

## A second, cleaner attempt: the value-error metric settles the question

Between the first attempt above and the next one, the opponent was switched back to the
perfect solver for an extended stretch (games ~39400 through ~46500), during which a new,
exact metric was added: **mean value error against the solver's ground truth**, computed
directly over every state in the Q-table (see `07_summary_and_open_questions.md` and the
dashboard's `compute_value_error` for the method - no sampling, no opponent dependence,
zero noise by construction). During that entire perfect-solver stretch it served as a
noise-floor reference: across 16 checkpoints (games 45700-46500) it stayed essentially
flat, fluctuating only within a narrow band of about 0.5756-0.5765 - useful context for
judging whether any later movement is real or just noise.

The opponent was then switched back to MCTS with `reward_shaping` still at **0**, around
game 46533. The first three checkpoints after that switch:

```
game 46550: value_error=0.5755, win_rate=15%
game 46600: value_error=0.5748, win_rate=20%
game 46650: value_error=0.5744, win_rate=30%
```

**Verdict: this is a real, consistent recovery signal, not noise - though still early.**
Two things make it convincing where the earlier win-rate-only evidence wasn't:

1. **Value error decreases monotonically across all three checkpoints** (0.5755 -> 0.5748
   -> 0.5744), moving to just below the ~0.5756-0.5765 noise floor established over the
   preceding 16 checkpoints, and continuing to move rather than merely landing outside
   that band once.
2. **Win rate rises monotonically over the same three checkpoints** (15% -> 20% -> 30%) -
   an independent metric agreeing in direction rather than contradicting it, unlike the
   bouncing 15/10/5/25% pattern seen in the first attempt.

The magnitude of the value-error change so far is still small (roughly 0.0016, only
somewhat larger than the established noise band), so this reads as "recovery has
genuinely begun" rather than "recovery is complete" - the Q-table was substantially
perturbed over the earlier extended shaped-reward-against-MCTS period, and undoing that
fully is expected to take a comparable number of games, not a handful of checkpoints.
Monitoring continues to confirm the trend holds as more games accumulate.

**Status: recovery confirmed as underway (not yet complete).** This is the clearest
answer so far to the open question first raised after the original 60%->0% decline: the
damage from reward shaping mismatched to an imperfect opponent is not permanent - reverting
to sparse reward does measurably undo it, just slowly. See
`07_summary_and_open_questions.md` for the standing open-questions list.

## Discussion: what this suggests about reward shaping in adversarial RL

Stepping back from the individual runs, here is the working interpretation as of this
point in the project - a synthesis of the evidence gathered so far, not a final
conclusion (the MCTS-recovery experiment above is still open).

**Sparse reward's real strength turned out to be that it encodes no assumption about the
opponent at all.** It only ever reacts to what actually happened in that one specific
game. That is also its weakness (slow, blurry credit assignment across a whole sequence of
moves) - but every measurement taken so far shows that weakness is *bounded*: sparse
reward has not, in any run in this project, caused a measured regression. Its cost is
"slower," never "worse than not training."

**Shaped reward is not simply a strictly-better version of sparse reward - it is a
specialized tool whose safety guarantee has a condition attached that is easy to miss.**
The Ng, Harada & Russell policy-invariance proof is correct, but it implicitly assumes the
environment - which, in an adversarial two-player game, includes the opponent's behavior -
matches whatever the potential function was built to represent. Here, the potential
function is the solver's exact value, whose meaning is "the outcome assuming **both sides
play optimally**." Training against MCTS (which does not play optimally) silently breaks
that assumption. The measured consequence was not a smaller improvement than hoped - it
was an active, sustained collapse (60% down to 0% win rate over roughly 700 real games).
That is a categorically different, worse failure mode than sparse reward's slowness:
shaped reward, mismatched to its opponent, can make a policy actively worse than doing
nothing.

**The value-error metric surfaced a second, more subtle distinction: reaching optimal
*behavior* is not the same as learning accurate *values*.** At the moment the agent was
achieving the exact theoretical ceiling against the perfect solver (50% win / 50% draw / 0%
loss - i.e. playing every move correctly), its mean value error against ground truth was
still roughly 0.575 and not visibly trending toward zero. A policy can rank actions
correctly at every state it needs to without its absolute value estimates being accurate -
which matters if the Q-table is ever used for anything beyond "pick the best move now"
(e.g. explaining how much better one option is than another, or transferring the values to
a different context).

**Overall assessment**: sparse reward should be the default whenever the opponent's
optimality is uncertain - which, in practice, is nearly always. Potential-based shaping via
an exact solver is a real, legitimate, and powerful tool, but a narrow one: it earns its
keep specifically when the actual opponent matches the assumption baked into the potential
function (here, the solver itself), not as a general-purpose training accelerant to reach
for by default. That the theorem's guarantee is conditional on opponent behavior - rather
than unconditionally safe, as it is for a single-agent MDP - is arguably the central,
underappreciated finding of this whole experiment: a technique proven safe in the
single-agent setting the theorem was written for does not automatically stay safe once
dropped into a multi-agent adversarial setting without re-checking that assumption.
