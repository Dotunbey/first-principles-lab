# The Perfect Solver: An Exact Solution, and a Soundness Bug Found the Hard Way

## Motivation

MCTS at any iteration count is still an *approximation* - it estimates values through
sampling. Since the reachable state space here is small (~4,030-4,974 states, see
`01_game_definition_and_state_space.md`), it's possible to compute the *true* value of
every position directly. That is a categorically different, and strictly stronger,
kind of "smart" than "MCTS with more iterations": not an approximation that gets better
with more compute, but the actual mathematically correct answer, computed once.

Before committing to this, a web search was done to check whether "Three Men's Morris" is
a known solved game and what the known result is. The search returned conflicting claims
(one source: a forced draw; another: a forced first-player win) from sources of uncertain
reliability (a couple were generic SEO content, not real game-theory references) - and in
any case, other people's stated results are for whatever exact ruleset *they* used
(diagonals or not, a "flying" endgame phase or not, etc.), which may not match the specific
rules implemented here (see `01_game_definition_and_state_space.md`). Rather than rely on
an unverifiable secondhand claim, the decision was to compute the answer directly for this
project's own exact rules.

## First attempt: memoized recursive negamax (unsound - discovered empirically)

The first implementation used a standard memoized negamax search, with one addition to
handle the fact that the sliding phase can revisit the same board position (that's what
the threefold-repetition rule is for) - meaning the state graph has **cycles**, which a
plain memoized recursive search isn't built to handle.

The fix attempted: track a `path` set of states currently open on the current line of
recursion; if the search revisits one of those, treat that branch as a draw (0) and stop
recursing - meant to mirror the idea that "if you can only get somewhere by looping,
neither side forced anything better."

Running this solver from the initial position gave:

```
Solved in 0.22s, 4974 distinct positions evaluated.
Result: a DRAW with perfect play by both sides.
```

### Validating the result - and finding it was wrong

Rather than trust this, the solver was pitted against MCTS (300 iterations/move, heuristic
rollout) over 20 games, alternating which side the "perfect" solver played. A truly optimal
player can never lose - only win or draw, by definition.

**Result: 6 wins, 5 draws, 9 losses.** The "perfect" solver lost nearly half its games.
This is logically impossible for a correct implementation, and revealed that the
initial approach was unsound, not just imprecise.

### Root cause

The path-based cycle-cutting heuristic corrupts the memoization cache. A state's computed
value can depend on *which DFS path happened to reach it first* (specifically, whether that
path's exploration order happened to trigger a "treat this as a draw" cutoff due to an
unrelated ancestor state coincidentally matching). That value then gets cached and reused
globally, including later when the same state is reached via a completely different,
non-cyclic path where the cutoff was never actually appropriate. In short: the technique
conflates "value of this state" (which should be a pure function of the state alone) with
"value of this state given the specific search path taken to reach it" (which is not the
same thing once memoization is involved).

## Fix: value iteration over the fully enumerated state graph

Rewritten (`perfect_solver.py`) to:

1. **Enumerate every reachable state via breadth-first search** from the initial position,
   recording each state's legal moves and where each one leads (either a terminal outcome,
   or another state).
2. **Value iteration**: initialize every non-terminal state's value at 0 (a neutral,
   "provisionally a draw" starting guess). Repeatedly sweep over every state, recomputing
   its value as the best over its available moves of (terminal value if that move ends the
   game, else the negation of wherever it leads), until a full sweep produces no changes
   anywhere (a fixed point).

This correctly handles cycles by construction - a state that's part of a cycle simply
settles at whatever value the fixed-point computation converges to (typically 0, a genuine
draw), rather than depending on which order a DFS happened to visit things in. This is the
same general technique used in real endgame-tablebase solvers for games with draws by
repetition.

### Re-solving

```
Enumerating every reachable position and solving via value iteration...
Solved 4974 distinct positions in 0.32s (9 sweeps to converge).
Result: X (the first mover) can FORCE A WIN with perfect play.
```

Note this **differs** from the first (buggy) attempt's answer of "draw" - confirming the
earlier result was simply incorrect, not just imprecisely worded.

### Re-validating against MCTS

Same test as before (20 games vs. MCTS at 300 iterations, alternating sides):

**Result: 12 wins, 8 draws, 0 losses.** Zero losses, exactly as a correct implementation of
an optimal player must produce. The specific pattern also matches the solved value exactly:
every game where the solver played X, it won (consistent with X being able to force a win);
every game where it played O, it drew (consistent with O's best possible outcome against
perfect play being a draw, never better).

## Why this episode matters methodologically

This is a case where a plausible-sounding, easy-to-justify implementation technique (path-
based cycle detection) produced a confident, clean-looking wrong answer (0.22 seconds, a
tidy "draw" result) that would not have been caught by code review or by "does the number
look reasonable" - it was only caught by treating the solver's own claim of optimality as a
falsifiable prediction and empirically testing it against an independent opponent. The
general lesson generalizes beyond this specific bug: for any system claimed to be optimal
or unbeatable, the claim should be checked by trying to beat it, not just by inspecting the
code that produces it.

## Implementation notes

- `state_key(game) = (tuple(game.board), game.phase, game.current_player)` - same state
  representation convention as the Q-learning agent's `hash_state`.
- `negamax_value(game)`: value from the perspective of whoever is about to move
  (+1 = can force a win, -1 = will lose to optimal play, 0 = draw).
- `best_move(game)`: picks the move maximizing the negated value of the resulting state
  (prefer a forced win, then a draw, then the least-bad loss).
- Both are backed by a module-level, lazily-computed cache (`_ensure_solved()`) - the
  expensive part (enumeration + value iteration) runs once, on first use; every subsequent
  call is an instant dictionary lookup.
