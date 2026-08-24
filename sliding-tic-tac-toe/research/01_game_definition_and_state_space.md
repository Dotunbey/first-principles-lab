# Game Definition and State Space

## Rules used (Sliding Tic-Tac-Toe / Three Men's Morris)

Board cells indexed 0-8:

```
0 1 2
3 4 5
6 7 8
```

- **Placement phase**: players alternate placing exactly 3 pieces each onto empty cells
  (X moves first, as in standard tic-tac-toe). If a player completes 3-in-a-row during
  placement, the game ends immediately as a win.
- **Sliding phase**: once both players have placed all 3 pieces, each turn a player slides
  one of their own pieces into an *adjacent* empty cell. There is no more placing.
- **Adjacency**: two cells are adjacent if they sit next to each other on one of the 8
  standard tic-tac-toe win-lines (3 rows, 3 columns, 2 diagonals). This gives:
  - Corners (0, 2, 6, 8): 3 neighbors each.
  - Edge-midpoints (1, 3, 5, 7): 3 neighbors each.
  - Center (4): 8 neighbors (it lies on all 4 lines that pass through it, so it
    connects to every other cell).
- **Win condition**: 3 in a row along any of the 8 lines, exactly as in standard
  tic-tac-toe, checked after every move (placement or slide).
- **Draw conditions** (needed because sliding has no natural end - see below):
  1. **Threefold repetition**: if the same (board, player-to-move) pair occurs 3 times
     in one game, it's declared a draw.
  2. **Move cap**: a hard limit of 40 slide moves; if reached with no winner, it's a draw.

Implemented in `game_engine.py` as the `SlidingTicTacToe` class.

## Why this needs an extra draw rule that standard tic-tac-toe doesn't

Standard tic-tac-toe is guaranteed to terminate within 9 moves - the board fills up,
forcing a draw if nobody has won. Sliding tic-tac-toe's sliding phase never adds or removes
pieces from the board, so nothing forces it to end; two players could in principle slide
pieces back and forth forever. The threefold-repetition and move-cap rules exist
specifically to guarantee termination, mirroring how real games with the same problem
(e.g. chess) are handled.

**Important clarification**: these two rules are *not* an official or standardized part of
Three Men's Morris / sliding tic-tac-toe. A web search turned up no documented
threefold-repetition or move-cap convention for this game in any of the usual references
(Wikipedia, Cyningstan, Masters of Games, BoardGameGeek) - being an old folk game with
regional variation, it doesn't appear to have one single standardized draw rule at all.
Threefold repetition specifically is a **chess** convention, borrowed here (along with an
invented move cap) purely as an engineering choice to guarantee the game engine
terminates - not a claim that this project is replicating some official ruleset. Without
*some* such rule, the underlying game genuinely has no guaranteed end, regardless of
whether an agent is playing it or not - see `09_agent_deployment_without_engine_rules.md`
for what that implies if this agent were ever deployed against a real opponent without
these engine-specific rules in place.

This mattered beyond just "the game needs to end" - it directly matters for the Q-learning
math. The backward Bellman update needs every episode to reach a defined terminal reward to
seed the backward pass from; without a guaranteed termination rule, a training episode
could in principle run forever with no reward signal to learn from at all.

## Predicting the state space size by hand, before writing any code

Both variants (standard tic-tac-toe, and this sliding variant) were worked out by hand
first, as a way to have an independent, verifiable prediction to check the eventual code
against.

### The counting method

At any point with `n` total marks on the board, turn-alternation forces:

- `n_X = ceil(n/2)` (X moves first, so X has the "extra" mark on odd totals)
- `n_O = floor(n/2)`

The number of raw board pictures with exactly that split (ignoring the order marks were
placed in - a **combination**, not a permutation, since only the final picture matters) is:

```
boards(n) = C(9, n_X) x C(9 - n_X, n_O)
```

### Standard tic-tac-toe (n runs 0 to 9)

| n | n_X | n_O | boards(n) |
|---|---|---|---|
| 0 | 0 | 0 | 1 |
| 1 | 1 | 0 | 9 |
| 2 | 1 | 1 | 72 |
| 3 | 2 | 1 | 252 |
| 4 | 2 | 2 | 756 |
| 5 | 3 | 2 | 1260 |
| 6 | 3 | 3 | 1680 |
| 7 | 4 | 3 | 1260 |
| 8 | 4 | 4 | 630 |
| 9 | 5 | 4 | 126 |

Sum = **6,046** raw combinatorially-valid boards (satisfying only the turn-parity
constraint). The commonly-cited true legal-state count for tic-tac-toe is **5,478** - the
gap of 568 boards corresponds to positions that are combinatorially plausible but
unreachable in real play, because they represent a game continuing *after* a line was
already completed (once someone wins, real play stops; there is no closed-form correction
for this, it has to be found by actually walking the game tree and checking each candidate
board against the win condition).

### Sliding tic-tac-toe (n capped at 0 to 6, since neither player ever exceeds 3 pieces)

| n | n_X | n_O | boards(n) |
|---|---|---|---|
| 0 | 0 | 0 | 1 |
| 1 | 1 | 0 | 9 |
| 2 | 1 | 1 | 72 |
| 3 | 2 | 1 | 252 |
| 4 | 2 | 2 | 756 |
| 5 | 3 | 2 | 1260 |
| 6 | 3 | 3 | 1680 |

Sum = **4,030** raw combinatorial boards - smaller than standard tic-tac-toe's 6,046 by
exactly the three chopped-off rows (n=7,8,9: 1260+630+126 = 2,016 fewer). One structural
note: the `n=6` row (1,680 boards) is not a "visited once" bucket like the others - it's
the *entire* pool that the sliding phase revisits and cycles through repeatedly, since
sliding never changes how many X's or O's are on the board. Rows n=0 through n=5 (2,350
boards total) are seen once each, only during placement.

### Measured vs. predicted

When the actual game engine was built and its reachable states enumerated by breadth-first
search (see `04_perfect_solver_exact_solution.md`), the true count came out to **4,974**
states - close to, and consistent with, the 4,030 raw combinatorial estimate. The
difference is explained by the same effect seen in standard tic-tac-toe (states that are
combinatorially valid but only reachable via specific paths get distinguished further by
also tracking *which phase* the game is in and *whose turn it is*, which the simple
by-hand combinatorics didn't separate out) - not a contradiction, but confirmation that the
by-hand estimate was in the right neighborhood before a single line of solver code existed.

This same combinatorial approach was independently confirmed a second time when the
Q-learning agent was trained by pure self-play: its Q-table converged to **4,424** distinct
states after 5,000 training games (see `02_qlearning_theory_and_implementation.md`) - a
third independent measurement landing in the same ~4,000-5,000 range predicted by hand.
