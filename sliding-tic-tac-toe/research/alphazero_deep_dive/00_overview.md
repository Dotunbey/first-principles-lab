# A Mini-AlphaZero, Built From Scratch: Complete Technical Deep Dive

## Who this is for

This is written for someone who has never seen AlphaZero's internals before, and wants to
actually understand every piece of it - not just "it uses a neural network and search,"
but the literal math, the literal code structure, and the mistakes that were made and
fixed along the way. No prior deep learning background is assumed; every formula is
explained in plain language before it's written symbolically.

This deep dive documents a real, working, from-scratch (numpy only, no PyTorch/TensorFlow)
implementation built for Sliding Tic-Tac-Toe, a small enough game that every single
position could be exactly solved (see `../04_perfect_solver_exact_solution.md`) - which
turned out to be an enormous advantage for building and debugging this system, since
almost nobody gets to check their neural network's answers against a mathematically
perfect oracle the way this project could.

## Why build this at all

Earlier work in this project (documented in `../06`, `../07`, `../08`) built a Q-learning
agent that stores one number per (state, action) pair in a plain lookup table. That
approach works, but has a hard ceiling: a table can only ever answer questions about states
it has *exactly* visited before. Ask it about a position it's never seen, and it has
nothing to offer.

AlphaZero's actual approach - a neural network, guided tree search, and a self-play loop
that continuously trains the network on the search's own output - solves this by
generalizing: the network can make a sensible guess about a position it has never seen,
based on similarity to positions it has. This document explains, in full, how that works
and how it was built here.

## The three pillars, and how they fit together

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
        ┌───────────────────────┐                             │
        │   1. NEURAL NETWORK   │                             │
        │  (value + policy)     │                             │
        │  neural_net.py        │                             │
        └───────────┬───────────┘                             │
                    │ guides                                  │
                    ▼                                         │
        ┌───────────────────────┐                             │
        │  2. GUIDED MCTS       │                             │
        │  (PUCT search)        │                             │
        │  neural_mcts.py       │                             │
        └───────────┬───────────┘                             │
                    │ generates games                         │
                    ▼                                         │
        ┌───────────────────────┐                             │
        │  3. SELF-PLAY LOOP    │                             │
        │  alphazero_selfplay.py│                             │
        └───────────┬───────────┘                             │
                    │ produces training data                  │
                    └─────────────────────────────────────────┘
                       (trains the network - the loop repeats)
```

**Pillar 1 - the network** (covered in `01_neural_network_architecture.md`): a small
hand-written neural network with two "opinions" about any position - how good it is
(the value head), and which moves look promising (the policy head).

**Pillar 2 - guided search** (covered in `02_mcts_with_puct.md`): Monte Carlo Tree Search,
same family of algorithm as the plain version built earlier in this project
(`../03_mcts_theory_and_implementation.md`), but now the network's opinions bias which
branches get explored and replace the need to simulate random rollouts to evaluate a
position.

**Pillar 3 - the self-play loop** (covered in `03_self_play_training_loop.md`): the closed
cycle that makes this whole thing self-improving. The network guides the search; the
search's own conclusions (which moves it ended up favoring, and what actually happened by
the end of the game) become the network's next training examples; a better-trained network
makes the next round of search stronger; repeat.

Two more documents round this out:
- `04_bugs_lessons_and_limitations.md` - two real bugs found during this build, and one
  fix attempt that only partially worked, each explained as a full case study, not just
  "here's the corrected code."
- `05_results_summary.md` - the actual numbers this system produced, compared against
  earlier approaches in this project, and what's still open.

## How to read this

Read in order (00 → 01 → 02 → 03 → 04 → 05) if you want the full build-up. If you already
understand neural networks and just want the MCTS/self-play specifics unique to this
project, skip to `02`. If you just want the headline results, jump to `05`.

## A note on notation used throughout

- `s` = a state (a board position, together with whose turn it is and which phase of the
  game - placement or sliding - it's in).
- `a` = an action (a move).
- Values are always on a **[-1, +1]** scale, and always **from the perspective of whoever
  is about to move** in that state: +1 means "the player to move can force a win," -1
  means "the player to move will lose to correct play," 0 means a draw. This exact
  convention is used everywhere in this project - the solver, the plain-table Q-learning
  agent, plain MCTS, and now the network - specifically so that every piece of this system
  can be directly compared against every other piece on the same scale.
