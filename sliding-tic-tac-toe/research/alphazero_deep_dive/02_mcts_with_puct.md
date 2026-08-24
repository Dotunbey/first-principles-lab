# Guided MCTS: PUCT Search, Fully Explained

Code: `neural_mcts.py`. Read `01_neural_network_architecture.md` first - this document
assumes the network from that file already exists and can answer "how good is this
position, and which moves look promising" for any given board.

## 1. Recap: what plain MCTS already does

This project already built and validated plain Monte Carlo Tree Search
(`../03_mcts_theory_and_implementation.md`, code in `mcts.py`), which repeats four steps
many times per real move:

```
1. SELECTION    - walk down the tree, picking the most promising child each time,
                  using the UCB1 formula: win_rate + c * sqrt(ln(parent_visits) / child_visits)
2. EXPANSION    - once an unexplored move is reached, add it as a new node
3. SIMULATION   - play the rest of the game out with random-ish moves, see who wins
4. BACKPROPAGATION - carry that result back up every node visited on the way down
```

The guided version in this document keeps steps 1 and 4 (selection and backpropagation),
but changes them to use the network, and completely replaces step 3 (no more random
simulated games at all).

## 2. What changes, at a glance

| | Plain MCTS (`mcts.py`) | Guided MCTS (`neural_mcts.py`) |
|---|---|---|
| Selection formula | UCB1 (win rate + generic exploration bonus) | **PUCT** (adds the network's move preference) |
| Expansion | One untried move at a time | **All legal moves at once**, each given a prior probability |
| Leaf evaluation | Simulate an entire rollout game with a hand-written heuristic | **One network call** - instant value estimate, no simulation |
| Backpropagation | Win (+1) / draw (+0.5) / loss (0) per visit | A continuous value in [-1, 1], flipped in sign at every level |

## 3. The PUCT selection formula

Plain UCB1, for reference:

```
UCB1(child) = (child's average win rate)  +  c * sqrt( ln(parent visits) / child visits )
              \_________________________/     \____________________________________/
                    "exploitation"                      "exploration"
```

**PUCT** (Predictor + Upper Confidence bound applied to Trees) adds a third ingredient -
the network's own opinion about how promising this move looked, *before* any search had
even happened:

```
PUCT(child) = Q(child)  +  c_puct * P(child) * sqrt(parent visits) / (1 + child visits)
              \______/     \_______________________________________________________/
           "exploitation"                    "exploration, weighted by prior"
```

- **`Q(child)`**: the average value backed up through this child so far (more on the sign
  convention below) - "based on what the search has actually found by looking here, how
  good does this move seem?"
- **`P(child)`**: the network's **policy prior** for this specific move - "before doing
  any search at all, how promising did the network's policy head think this move was?"
- **`c_puct`**: a constant (1.5 in this implementation) controlling how much weight the
  prior gets versus the accumulated search evidence.
- **`sqrt(parent visits) / (1 + child visits)`**: this shrinks as a specific child gets
  visited more, so the influence of the prior naturally fades as real search evidence
  accumulates - early on, trust the network's hunch; later, trust what's actually been
  discovered.

This is the single most important difference from plain UCB1: instead of treating every
untried move as equally worth a first look, moves the policy network already favors get
explored preferentially - dramatically narrowing the search compared to blind UCB1,
provided the network's opinion is any good at all.

## 4. Expansion: all children at once, with priors

Plain MCTS expands one untried move per visit to a node. Guided MCTS instead expands an
entire node in a single step, because a single network call already produces a
probability for *every* legal move simultaneously - there's no reason to space that out:

```
def _expand(node, net):
    value, policy_probs = net.predict(board, phase, current_player)   # ONE network call

    for each legal move:
        prior = policy_probs[that move's index]         # masked + renormalized over legal moves only
        create a child node, storing that prior

    return value      # used immediately for backpropagation - see below
```

Diagram of one expansion:

```
                    BEFORE expansion                          AFTER expansion
                                                        (all children created at once,
                                                         each tagged with its own prior)

         ┌─────────┐                              ┌─────────┐
         │  LEAF    │                              │  NODE    │
         │ (unex-   │        one call to           │(expanded)│
         │ panded)  │ ───── net.predict() ───────▶ └────┬────┘
         └─────────┘                                    │
                                                  ┌───────┼───────┬─────────┐
                                                  ▼       ▼       ▼         ▼
                                              move A   move B  move C   move D
                                              P=0.51   P=0.30  P=0.12   P=0.07
                                             (child) (child) (child)  (child)
```

## 5. Evaluation: the value network replaces the rollout entirely

Plain MCTS's step 3 plays an entire simulated game to its conclusion (using a hand-written
heuristic) just to get a single win/loss/draw sample. Guided MCTS skips this completely:
the value the network predicts *is* the evaluation, used directly. This is both much
faster (one forward pass through a small network vs. simulating up to dozens of moves) and,
if the network is any good, more accurate than a single noisy random-ish playout - measured
directly in this project at 5x faster for equivalent iteration counts (see
`05_results_summary.md`).

## 6. Backpropagation: the sign-flip convention (and the bug that broke it)

Every node's stored value means "average outcome, **from the perspective of whoever is
about to move at that specific node**." Since players alternate, a value that's good news
one level down the tree is bad news one level up - so backpropagation flips the sign at
every step:

```
def _backup(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value          # flip: one level up is the OPPONENT's perspective
        node = node.parent
```

Diagram, tracing a value of `+0.8` (great for whoever's about to move at the leaf) up
through three levels:

```
   ROOT (X to move)                     value received here: -(-(0.8)) = +0.8 (good for X - correct, since ancestor of ancestor of leaf shares X's perspective)
        │
        ▼
   CHILD (O to move)                    value received here: -(0.8) = -0.8 (bad for O - correct, X is winning)
        │
        ▼
   LEAF (X to move, just evaluated)     value = +0.8 (from the network, "good for X")
```

### The bug this project actually hit

`game_engine.py`'s move-application code **returns early on a winning move, before
flipping whose turn it is**. So immediately after a win, `current_player` still names the
*winner*, not "whoever's turn would be next." The first version of `_terminal_value()`
assumed that field always flips and used it directly - silently **inverting every single
win and loss** it encountered at a terminal node.

The measured consequence, before any code was even reviewed line-by-line, was almost
comedic: pitted against a random opponent, guided MCTS **lost 14 games out of 20** -
worse than random play, which should be close to impossible for a functioning search.
That result was the signal something was fundamentally broken (see
`04_bugs_lessons_and_limitations.md` for the full debugging story and the fix).

The fix: track **who actually moved to create this node** explicitly (`player_just_moved`,
the same pattern the original plain `mcts.py` already used, which had never hit this bug
because it never relied on the post-move `current_player` field for exactly this reason),
and compute the terminal value relative to that explicit identity instead of trusting the
engine's field.

## 7. Choosing the real move: visit counts, not raw values

After running many iterations (each one of them a full
selection → expansion → evaluation → backpropagation cycle), the actual move played is
whichever child was visited the *most* - not necessarily the one with the single highest
average value. This is a deliberate, standard choice (called the "robust child" in MCTS
literature): a move visited thousands of times with a slightly lower average is generally
more trustworthy than one visited only a handful of times with a slightly higher average,
because the search spent more total effort confirming the former.

## 8. Two additions specific to self-play: temperature and Dirichlet noise

These aren't used when the search is just being asked "what's the best move" for a single
decision - they're specifically for generating *diverse* self-play training data (covered
fully in `03_self_play_training_loop.md`), but the mechanics belong here:

**Temperature** reshapes the visit-count distribution before it's used to actually choose
a move:

```
policy(move) = visit_count(move) ^ (1 / temperature)   ... then renormalized to sum to 1
```

- `temperature = 1.0`: use the raw visit counts as-is - more randomness, more diverse games.
- `temperature → 0`: sharpens the distribution toward the single most-visited move -
  effectively deterministic, "always play the best move found."

This project uses `temperature = 1.0` for the first 6 moves of every self-play game (for
opening diversity) and `temperature = 0.1` afterward (increasingly close to deterministic
"just play the best move" for the rest of the game).

**Dirichlet noise** is mixed into the *root* node's priors only, specifically during
self-play (never during a "just give me the best move" evaluation):

```
child.prior = child.prior * (1 - frac)  +  dirichlet_noise_sample * frac
```

(`frac = 0.25` in this implementation.) The purpose: without this, self-play exploration
depends entirely on the network already being right about which moves are interesting -
which is circular early in training, when the network doesn't know anything yet. Injecting
random noise into the root's priors guarantees every legal move gets *some* chance of
being explored, regardless of what the (possibly still-bad) network currently believes.

## What's next

`03_self_play_training_loop.md` covers how this search gets used to actually generate
training data, and how that data trains the network that guides the next round of search.
