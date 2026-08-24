# Variant Registry - All Trained Agents, Ready for a Future Tournament

A running roster of every distinct trained agent produced across this project, with
exactly what it is, how it was trained, and how to load it. Kept up to date as new
variants are built. When the tournament happens, this is the source of truth for who's
competing.

## Neural network variants (the AlphaZero-style build)

All share the same architecture (`neural_net.py`, `TicTacToeNet`, 64 hidden units) and are
loaded the same way:

```python
from neural_net import TicTacToeNet
from neural_mcts import neural_mcts_search

net = TicTacToeNet(hidden_size=64, seed=42)
net.load('<filename>.pkl')
move, root = neural_mcts_search(game, net, iterations=100)
```

| # | Name | File(s) | What's different | Status |
|---|---|---|---|---|
| 1 | **unbalanced** | `selfplay_trained_net.pkl` | The original run - fresh examples only each iteration, no rebalancing, no teacher | Finished (300 iter) |
| 2 | **balanced** | `selfplay_trained_net_balanced.pkl` | + 8,000-example replay buffer, + X/O rebalancing in training batches | Finished (300 iter) |
| 3 | **teacher** | `selfplay_trained_net_teacher.pkl` | + perfect solver plays O's actual moves during self-play generation | Finished (300 iter) |
| 4 | **exploration** | `selfplay_trained_net_exploration.pkl` | Same as balanced, but `temperature_moves=20` (up from 6) and `noise_frac=0.4` (up from 0.25) | Finished (300 iter) - decisive test: X 20/0, O 0/20 |
| 5 | **o_compute** | `selfplay_trained_net_o_compute.pkl` | Same as unbalanced, but O gets 2x MCTS iterations on its own turns | Finished (300 iter) - decisive test: X 20/0, O 0/20 |
| 6 | **twonet** | `selfplay_trained_net_twonet_X.pkl` + `_O.pkl` | Two completely separate networks (different seeds), one per role - no shared weights at all | Finished (300 iter) - decisive test: X 20/0, O 0/20 |
| 7 | **catchup** | `selfplay_trained_net_catchup_X.pkl` + `_O.pkl` | Same two-network split as twonet, but X's training is frozen (no gradient updates) whenever O falls >0.05 behind on value error | Finished (300 iter) - froze only 9/300 iters, gate tracks calibration not win rate; decisive test: X 20/0, O 0/20 |
| 8 | **imitate** | `selfplay_trained_net_imitate.pkl` | Same as teacher, but O's policy training target is one-hotted onto the solver's actual move (not just its played move) - the most targeted fix identified after variants 2-7 all failed the decisive test | Finished (300 iter) - decisive test: X 20/0, O 0/20 - **surprising negative result, see 04/05 docs** |
| 9 | **catchup_wr** | `selfplay_trained_net_catchup_wr_X.pkl` + `_O.pkl` | Same two-network split as catchup, but the freeze gate watches O's rolling win rate directly (hysteresis: freeze <15%, unfreeze >30%) instead of value-calibration gap | Running |

Also present but **not a real player** - useful only for the Stage 1 generalization
validation, its policy head was never trained (still random init): `value_only_net.pkl`.

Full technical detail on each: `alphazero_deep_dive/00` through `05`.

## Table-based Q-learning variants (the earlier, pre-network approach)

These live in `web_dashboard/`, are loaded via the dashboard's `QLearningAgent`
(`q_learning.py`), and represent a plain lookup table rather than a network - no
generalization to unseen states, but simpler and directly comparable to the neural
variants above.

| Name | File | What's different | Status |
|---|---|---|---|
| **mixed-history agent** | `live_session.pkl` | Trained across multiple reward schemes over its lifetime (sparse, then solver-shaped, with experiments switching back and forth) - see `06_reward_shaping_experiment.md` for the full history. Not a clean single-condition agent. | ~47,660 games |
| **pre-TD0 backup** | `live_session_backup_before_td0_experiment.pkl` | Snapshot of the mixed-history agent taken before the TD(0) experiments began | Frozen snapshot |
| **TD(0) vs strong MCTS** | `live_session_td0.pkl` | Online TD(0) updates, trained against MCTS at 400 iterations - the run that revealed the "drawing trap" | See `08_td0_online_learning.md` |
| **TD(0) vs weak MCTS** | `live_session_td0_easy_mcts.pkl` | Same as above but MCTS at only 50 iterations - shows real wins, escapes the trap | See `08_td0_online_learning.md` |
| **TD(0) self-play (table)** | `live_session_selfplay.pkl` | Table-based agent, TD(0) updates, plays both sides itself (no MCTS opponent at all) | ~2,105+ games at last check |
| **Original self-play Q-learner** | `q_learning_model.pkl` | The very first self-play agent built in this project, before the web dashboard existed (`q_learning.py`'s standalone script) | 5,000 games, 99.5% vs random |

## Non-learned baselines (useful as fixed tournament opponents/benchmarks)

| Name | Code | What it is |
|---|---|---|
| **Perfect solver** | `perfect_solver.py` (`best_move()`) | Exact, mathematically unbeatable - solved via value iteration over all ~4,974 states. The ultimate ground truth. |
| **Plain MCTS** | `mcts.py` (`mcts_search()`) | Heuristic-rollout MCTS, no learning, no network - configurable iteration count |
| **Random** | `random.choice(game.get_valid_moves())` | The baseline everything should beat easily |

## Tournament results (full round robin, greedy/deterministic, 132 games)

Every ordered pair of the 9 network variants + 3 baselines played once (each
matchup is fully deterministic with no exploration noise, so repeats would be
identical - one game per ordered pair is the complete result). Overall record
pooled across both sides played:

| Rank | Name | Wins | Losses | Draws |
|---|---|---|---|---|
| 1 | o_compute | 15 | 7 | 0 |
| 2 | twonet | 15 | 6 | 1 |
| 3 | unbalanced | 14 | 8 | 0 |
| 4 | exploration | 13 | 7 | 2 |
| 5 | catchup | 13 | 8 | 1 |
| 6 | balanced | 12 | 7 | 3 |
| 7 | perfect_solver | 12 | 9 | 1 |
| 8 | catchup_wr | 10 | 7 | 5 |
| 9 | teacher | 6 | 13 | 3 |
| 10 | imitate | 5 | 14 | 3 |
| 11 | plain_mcts | 5 | 12 | 5 |
| 12 | random | 0 | 22 | 0 |

Notable: **teacher and imitate rank near the bottom overall**, despite being
the two variants specifically designed to fix the X/O asymmetry via solver
involvement. This is a new finding not visible in the decisive-test-vs-solver
numbers alone (`05_results_summary.md`) - both were trained with the perfect
solver playing/labeling O's moves during self-play, which may have made their
policies more specialized to "defend against a perfect X" specifically,
at some cost to general competitiveness against a wider range of opponents
(other variants, plain MCTS). `o_compute`, `twonet`, and the original
`unbalanced` run - none of which involved the solver during training at all -
rank highest overall.

The solver's 9 losses were **all** as O, one to each network variant playing
X - consistent with, not contradicting, its "unbeatable" status: X has a
proven forced win (`04_perfect_solver_exact_solution.md`), so no O-side play,
however optimal, can avoid losing to a sufficiently strong X. Its only draw
was against `plain_mcts` playing X, which wasn't strong enough to force the
win. Full per-game results: `../tournament_results.csv`. Script:
`../tournament.py`.

## How this feeds the eventual tournament

Every entry above can play a game against any other by having each side call its own
`best_move`/`neural_mcts_search`/`choose_action` function on its turn - the game engine
(`game_engine.py`) is agent-agnostic, so mixing types (e.g. `twonet`'s O-network vs. the
`teacher` run's shared network vs. plain MCTS vs. the perfect solver) is straightforward.
When the tournament is run, results should reference agents by the **Name** column here,
and any new variant built after this point should be added as a new row before it's
allowed to compete.
