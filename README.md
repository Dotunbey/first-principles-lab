# First Principles Lab

**Reinforcement learning from scratch — no frameworks, no libraries, no shortcuts.**

This is a research lab for building game-playing AI agents entirely from first principles. Every component — the game engine, the Q-learning agent, the Monte Carlo Tree Search, the perfect solver, the neural network, the backpropagation, the Adam optimizer — was written by hand. No PyTorch. No TensorFlow. No Keras. No scikit-learn. Just NumPy and pure Python.

---

## The three experiments

### 1. Sliding Tic-Tac-Toe — the core research project

A custom game variant with two phases: **placement** (each side places 3 pieces) and **sliding** (pieces slide along rows/columns rather than being removed). The goal: build agents that learn to play it, then test what happens when you apply real RL techniques and let the measurements decide.

#### What was built

| Component | File | Result |
|---|---|---|
| Game engine | `game_engine.py` | 2,000 random self-play games — zero deadlocks, zero runaway games, max 46 moves (matches the design exactly: 6 placement + 40 slide-cap) |
| State-space prediction | `research/01` | Predicted **4,030** raw combinatorial boards by hand, before writing any code. Actual enumeration gave **4,974** reachable states |
| Q-learning agent | `q_learning.py` | 5,000 games of true self-play, Q-table grew to 4,424 states (matching the hand prediction), **99.5% win rate vs. random** |
| MCTS | `mcts.py` | UCB1, one bug found and fixed pre-launch (terminal nodes falsely reporting untried moves), validated **20/20 wins** vs. random with zero training |
| Perfect solver | `perfect_solver.py` | First version found **unsound** — it lost 9/20 games to MCTS despite claiming optimality. Root-caused to a memoisation/cycle-detection flaw, then corrected via value iteration over the fully enumerated state graph. Re-validated at **12 wins / 8 draws / 0 losses** vs. MCTS |
| Neural network | `neural_net.py` | Hand-written forward/backward pass, 2 hidden layers, ReLU, He init, Adam optimizer — no ML framework |
| AlphaZero self-play | `alphazero_selfplay.py` | Closed loop: network-guided MCTS generates training data, which trains the network that guides the next round |

#### Key finding: the solver says X can force a win

The corrected solver determined that under perfect play, **X can force a win** in sliding tic-tac-toe. The first, buggy version of the solver had incorrectly reported a draw. This was caught because the buggy solver lost to MCTS — a simple empirical test that would have been skipped in a "trust the math" approach.

#### Key finding: reward shaping fails against an imperfect opponent

This is the central result of the project.

**Potential-based reward shaping** (Ng, Harada & Russell, 1999) was implemented using the perfect solver's exact values as the potential function. The theory guarantees that shaping leaves the optimal policy provably unchanged.

- **Against a perfect solver**: the agent reached the exact theoretical ceiling — 50% win / 50% draw / 0% loss. The shaping worked.
- **Against MCTS (an imperfect opponent)**: win rate collapsed **60% → 0% over ~700 games**. The shaping actively made the agent worse.

The diagnosis: the policy-invariance guarantee holds only when the opponent matches the potential function's assumption — "both sides play optimally." MCTS does not play optimally, so the shaping misled the agent. This is a real, observed, opponent-dependent failure mode — not a bug in the implementation.

**The lesson**: reward shaping is a specialised tool with a condition attached that is easy to miss. It is not a general-purpose training accelerant. Sparse reward is the safer default whenever the opponent's optimality is uncertain — which, in practice, is nearly always.

#### Key finding: generalisation beyond memorisation

The AlphaZero value head was trained on **80%** of the solver's 4,974 exact states and tested on the **20% it never saw**. Test MAE went **0.566 → 0.206**. This is concrete proof that a neural network generalises to unseen states — something a Q-table, by construction, cannot do.

#### Key finding: X/O skill asymmetry

AlphaZero self-play produced a one-sided pattern that was not exploration noise. When both sides played fully greedy (no exploration), **X won 20/20, O won 0/20**. The shared network learned different skill levels for the two sides. This is a genuine learned imbalance, not a bug.

---

### 2. Tic-Tac-Toe Classic — the transfer test

The same research applied to standard 3×3 tic-tac-toe, with one extra question: **does anything transfer from the sliding game?**

#### Key finding: warm-starting actively hurts

The sliding game's placement phase produces state hashes that are **byte-identical** to classic tic-tac-toe's early-game states. So a Q-table trained on sliding tic-tac-toe should, in principle, give a classic agent a head start.

**It doesn't.**

Two matched dashboards were run: 5,013 games with the Q-table warm-started from the sliding agent, and 5,014 games fresh. Same hyperparameters. Same opponent (MCTS). Same seed.

The fresh agent was ahead at every checkpoint. At game 0, the inherited Q-values had a mean error of **0.5686** — worse than the fresh agent's **0.4275** at game 25. One position had an inherited value of **+0.863** (near-certain win) when the true classic value was **-1.000** (certain loss).

The matching was a **representation** match, not a **value** match. The same board, the same piece count, the same phase — but the game that continues from that state is different. The sliding game's placement value encodes what happens after pieces start sliding, which has nothing to do with what happens in classic tic-tac-toe from that same board.

**The lesson**: transfer learning requires value transfer, not just representation transfer. Matching state hashes is necessary but not sufficient.

---

### 3. Oware — the earliest work

Oware (Awari) is a Mancala-style game where both players race to score seeds in their scoring pit. The engine is built from scratch. The Q-learning agent uses true self-play plus replay of recorded expert games.

#### What was built

| Component | File | Notes |
|---|---|---|
| Game engine | `oware_engine.py` | Board state, valid moves, captures, game-over detection |
| Q-learning agent | `enhanced_learn_oware.py` | Self-play + expert game replay, persisted to disk |
| Reward system | `reward_system.py` | Non-potential-based shaping: +100 win, -100 loss, +50 per capture, +10 centre-house control |
| Minimax baseline | `oware_ai.py` | Minimax with alpha-beta pruning, depth 3, heuristic eval |

The Oware reward shaping is **not** potential-based — the capture and centre-control bonuses do not carry the Ng/Harada/Russell guarantee. They are a different, less rigorous category of shaping. This matters: the sliding-project finding about shaping failing against an imperfect opponent does not automatically apply here, but the same question applies — does the bonus assume something about the opponent that isn't true?

---

## What makes this different from a typical ML repo

**Negative results are kept.** Most ML repos only show what worked. This one documents what failed and why:

- The first perfect solver was unsound. It was caught by an empirical test, not a code review.
- Reward shaping collapsed the agent against MCTS. The failure was kept as a finding, not discarded.
- Warm-starting from the sliding agent actively hurt the classic agent. The failure was kept as a finding, not discarded.
- TD(0) had a real bug that shipped in the first version — only detected outcomes when the agent's own move ended the game. It was caught by inspecting the raw update log directly.

**Every claim is backed by a measurement from the actual run.** No "achieves state-of-the-art performance." No "significantly improves." Just: "99.5% win rate vs. random over 200 games" or "MAE 0.566 → 0.206 on the held-out 20%."

**The methodology is: build it, test it empirically, let the measurement overturn the hypothesis.** If the result is negative, the negative result is the finding.

---

## Project structure

```
first-principles-lab/
├── sliding-tic-tac-toe/          # Core research project
│   ├── game_engine.py            # Sliding variant engine
│   ├── q_learning.py             # Tabular Q-learning agent
│   ├── mcts.py                   # Monte Carlo Tree Search
│   ├── perfect_solver.py         # Exact solver (value iteration)
│   ├── neural_net.py             # Hand-written neural network
│   ├── neural_mcts.py            # MCTS guided by the neural net
│   ├── alphazero_selfplay.py     # AlphaZero self-play loop
│   ├── research/                 # 11 research documents
│   │   ├── 01_game_definition_and_state_space.md
│   │   ├── 02_qlearning_theory_and_implementation.md
│   │   ├── 03_mcts_theory_and_implementation.md
│   │   ├── 04_perfect_solver_exact_solution.md
│   │   ├── 05_reward_shaping_theory.md
│   │   ├── 06_reward_shaping_experiment.md
│   │   ├── 07_summary_and_open_questions.md
│   │   ├── 08_td0_online_learning.md
│   │   ├── 09_alphazero_deep_dive.md
│   │   ├── 10_alphazero_style_implementation.md
│   │   └── 11_results_summary.md
│   ├── play_vs_network/          # Flask dashboard — play vs. the trained network
│   ├── web_dashboard/            # Live training dashboard
│   └── selfplay_dashboard/       # Self-play monitoring dashboard
│
├── tic-tac-toe-classic/          # Classic variant + transfer test
│   ├── game_engine.py
│   ├── q_learning.py
│   ├── mcts.py
│   ├── perfect_solver.py
│   ├── neural_net.py
│   ├── neural_mcts.py
│   ├── alphazero_selfplay.py
│   ├── bootstrap_from_sliding_agent.py  # Warm-start experiment
│   ├── research/
│   │   └── warm_start_experiment.md
│   └── play_vs_network/          # Flask dashboard — play vs. the trained network
│
├── oware/                        # Oware/Awari project
│   ├── oware_engine.py
│   ├── learn_oware.py
│   ├── enhanced_learn_oware.py
│   ├── reward_system.py
│   ├── oware_ai.py               # Minimax baseline
│   └── README.md
│
├── snake_game.py                 # Snake (pygame)
├── tic_tac_toe.py                # Two-player tic-tac-toe (pygame)
└── blog/                         # Personal blog (Next.js)
```

---

## Running the code

### Play against the trained AlphaZero network (Classic Tic-Tac-Toe)

```bash
cd tic-tac-toe-classic/play_vs_network
python app.py
```

Open `http://127.0.0.1:5021` in your browser. You play X, the trained network plays O.

### Play against the trained network (Sliding Tic-Tac-Toe)

```bash
cd sliding-tic-tac-toe/play_vs_network
python app.py
```

### Run the full research pipeline

```bash
cd sliding-tic-tac-toe

# Build the perfect solver
python perfect_solver.py

# Train the Q-learning agent
python q_learning.py

# Train the AlphaZero network
python alphazero_selfplay.py --iterations 300
```

### Oware

```bash
cd oware
pip install -r requirements.txt
python enhanced_learn_oware.py
```

---

## Dependencies

- Python 3.11+
- NumPy (the only ML-related dependency — and even that is used for array operations, not machine learning)
- Flask (for the play dashboards)
- pygame (for the classic snake and tic-tac-toe)

---

## The methodology

Every experiment in this repo follows the same pattern:

1. **Predict by hand.** Work out what the system should do before writing any code.
2. **Build from scratch.** Write the engine, the algorithm, the optimizer — no framework.
3. **Test empirically.** Run it. Let the measurement decide.
4. **Let the result overturn the hypothesis.** If the result is negative, the negative result is the finding. Keep it.

The two documented failures — reward shaping collapsing against MCTS, and warm-starting actively hurting the classic agent — are the most valuable parts of this repo. They are specific, measured, and explained. They are kept in the codebase as recorded lessons, not discarded because they were inconvenient.

---

## Licence

MIT
