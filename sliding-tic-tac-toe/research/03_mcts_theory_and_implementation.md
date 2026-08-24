# Monte Carlo Tree Search: Theory and Implementation

## The core distinction from Q-learning

Q-learning is **memory-based**: train first (many games, updating a table), then at play
time just look up a stored value - instant, but requires training before it's any good.

MCTS is **search-based**: no pre-training at all. At the moment it's asked to move, it runs
many simulated random-ish playouts from the current position, entirely "in its head," and
picks whichever move those simulations liked best. It can play reasonably on move one, with
zero prior games played, as long as it knows the rules - the cost is spent at decision time
instead of beforehand.

## The four-step loop, repeated many times per real move

1. **Selection** - starting at the real current state, descend through the tree so far,
   at each level picking the child with the highest **UCB1** score, until reaching a node
   with an untried move (or a terminal state).

   ```
   UCB1 = (W_i / N_i) + c * sqrt( ln(N_total) / N_i )
   ```
   - `W_i / N_i`: this move's observed win rate so far.
   - `N_total`: total simulations run from the current real position so far.
   - `N_i`: how many times this specific move has been tried.
   - `c`: exploration constant (1.41, approximately sqrt(2), used throughout).

2. **Expansion** - add one new child node for an untried move.

3. **Simulation ("Monte Carlo") / rollout** - play the rest of the game out from that new
   node using some move-selection policy (see "rollout policy" below), all the way to a
   terminal result.

4. **Backpropagation** - walk back up the path just taken, incrementing `visits` (N) at
   every node, and `wins` (W) wherever the simulated result matches whichever player made
   that particular move (0.5 credit for a draw).

After many repetitions, the move actually played is whichever root-level child has the
**highest visit count** (not highest win rate - the "robust child" convention).

### Worked numeric example of UCB1 (done by hand)

Given 10 simulations distributed across 3 candidate moves, `c=1.41`, `N_total=10`:

| Move | N_i | W_i | Win rate | UCB1 |
|---|---|---|---|---|
| A (center) | 6 | 4 | 0.667 | 0.667 + 1.41*sqrt(ln10/6) = 1.541 |
| B (corner) | 3 | 1 | 0.333 | 0.333 + 1.41*sqrt(ln10/3) = 1.569 |
| C (edge)   | 1 | 0 | 0.000 | 0.000 + 1.41*sqrt(ln10/1) = 2.140 |

Move C wins the selection despite a 0% observed win rate, because it's only been tried
once - its exploration term is large. This is the mechanism that guarantees no move is
permanently starved just because its first sample looked bad: the exploration term shrinks
slowly overall (`ln(N_total)` grows very gradually) but shrinks quickly *per move* as that
move's own `N_i` grows.

## Relationship to AlphaGo / AlphaZero

For a game this small, pure MCTS is overkill (the whole tree could be solved exactly - see
`04_perfect_solver_exact_solution.md`). But the general architecture matters for larger
games: AlphaGo used MCTS as the overall framework but replaced the "blind" parts with
neural networks - a policy network biased the Selection step's exploration term toward
moves a strong player would consider, and rollouts were informed by a value network's
instant position estimate rather than being played out fully at random.

One correction made during this project to a secondhand description of this: the *original*
AlphaGo **blended** a value-network estimate with the result of a fast policy-guided
rollout (not a full replacement). It was **AlphaZero**, the later system, that removed
rollouts entirely and relied purely on the value network.

## Implementation (`mcts.py`)

- `TreeNode`: holds a game state, parent/children, `visits`, `wins`, and remaining
  `untried_moves`.
- `select_child()`: implements the UCB1 formula above.
- `expand()`: turns one untried move into a new child.
- `rollout(game_state)`: plays a copy of the game to completion (see rollout policy below).
- `backpropagate(winner)`: walks back to the root, updating visit/win counts.
- `mcts_search(root_state, iterations)`: runs the loop `iterations` times, returns the
  most-visited root move.

### A bug caught before first use

`get_valid_moves()` does not check `game_over` - it just reports legal moves based on board
state, regardless of whether the game has already been won. This meant a `TreeNode` that
had just become terminal would still report "untried moves," letting the search try to
expand past an already-finished game (and potentially call `make_move` on a game that had
already ended, corrupting state). Fixed by forcing `untried_moves = []` whenever
`game_state.game_over` is true, at node-construction time:

```python
self.untried_moves = [] if game_state.game_over else game_state.get_valid_moves()
```

This was found and fixed *before* the first real test run, by inspecting the interaction
between `get_valid_moves()` and node construction, rather than being discovered via a
failing test - worth noting as a contrast to the much more serious bug found later in the
perfect solver (`04_perfect_solver_exact_solution.md`), which was *not* caught by
inspection and only surfaced through empirical head-to-head testing.

## Initial validation

20 games, MCTS (X, 200 iterations/move, original random rollout) vs. a uniformly random
opponent (O): **20/20 wins**, 0 losses, 0 draws - confirming MCTS plays well immediately
with no training, as the theory predicts.

## Strengthening the rollout policy

The original `rollout()` chose moves uniformly at random during simulation. This was
upgraded to a **one-ply-lookahead heuristic** (`heuristic_rollout_move`):

1. If any legal move wins immediately, take it (never miss a free win).
2. Otherwise, check every legal move for whether it hands the opponent an immediate win on
   their very next turn; prefer moves that don't.
3. If no candidates are "safe" by that test, fall back to uniform random among all legal
   moves.

This is a "light/heavy playout" in MCTS terminology - it doesn't change what MCTS is doing
structurally, only how informative each simulated game's result is (a "smart-ish"
simulated game is more useful evidence than a purely random one).

### Cost of the stronger rollout

Benchmarked on the initial board position:

| Iterations | Random rollout | Heuristic rollout |
|---|---|---|
| 150 | 0.03s | 1.01s |
| 500 | 0.09s | 2.52s |
| 1,000 | 0.31s | 3.61s |
| 5,000 | 1.25s | (not re-measured after upgrade) |
| 20,000 | 4.71s | (not re-measured after upgrade) |

The heuristic rollout is roughly 30x slower per iteration (each simulated step now does a
small nested lookahead instead of one random draw), but produces much more informative
simulated outcomes per iteration - the intended tradeoff.
