# Introduction and Motivation

## Where this started

This research grew out of an earlier, separate project: an Oware (Awari) game-playing AI
built with tabular Q-learning. A review of that codebase surfaced several structural
problems that motivated everything that followed here:

- The Q-table was keyed on the exact board+score+player state, with no generalization
  across similar positions - viable only because the state space is small enough to
  enumerate, which raised the question of *how* small a game needs to be for this to work.
- Self-play was implemented against a *random* opponent rather than true self-play (the
  same policy playing both sides), which under-trains a policy against real resistance.
- A "reward shaping" design document existed (capture bonuses, center-control bonuses)
  but was never actually wired into the learning update - the code only used the
  win/loss/draw outcome. This planted the question of whether shaped rewards actually
  help, and whether they can ever introduce bias.
- An "expert game training" feature was a stub that hard-coded a fixed boost to Q-values
  for two hand-picked example games, rather than genuinely replaying real recorded
  expert games through the engine.

These were fixed in the Oware project (reward shaping wired into a proper backward
discounted return, true self-play with epsilon decay, real expert-game replay), and that
work is documented separately in that project's own files. But it left an open question
that Oware itself is too large to answer directly: Oware's state space is far too big to
solve exactly, so there was no way to check whether the learned Q-values were actually
*correct*, only whether they seemed to work.

## Why Sliding Tic-Tac-Toe

To get a testbed small enough to fully verify claims about correctness - not just "does it
seem to play well" but "is this provably the right answer" - this project moved to a much
smaller game: **Sliding Tic-Tac-Toe**, also known as **Three Men's Morris**. It keeps the
two core ideas from Oware (tabular Q-learning, self-play) but is small enough that:

1. The exact size of the reachable state space can be derived by hand with combinatorics
   (worked out before writing any code - see `01_game_definition_and_state_space.md`).
2. The game can be **solved exactly** via value iteration over the full state graph (see
   `04_perfect_solver_exact_solution.md`), giving a provably optimal reference point that
   Oware could never provide.
3. Every claim made along the way - "the agent is learning," "MCTS is smarter now,"
   "this reward scheme is unbiased" - could be checked against real, measured data rather
   than taken on faith.

## What this collection of documents covers

The work proceeded in stages, each documented in its own file:

- `01_game_definition_and_state_space.md` - the exact rules used, and the combinatorics
  that predict the state space size before any code exists.
- `02_qlearning_theory_and_implementation.md` - the Q-learning math (Bellman update,
  epsilon-greedy exploration) and how it was implemented and validated.
- `03_mcts_theory_and_implementation.md` - Monte Carlo Tree Search (UCB1, the four-step
  loop), its implementation, a bug found before first use, and a later strengthening of
  its rollout policy.
- `04_perfect_solver_exact_solution.md` - building an exact solver, a serious soundness
  bug discovered only through empirical testing (the solver initially *lost* games it
  should have been mathematically incapable of losing), and the fix.
- `05_web_dashboard_and_engineering_challenges.md` - the live training dashboard built to
  make every part of training observable and controllable, and two real production bugs
  encountered and fixed along the way (a silent-overwrite UI bug, and a genuine thread
  freeze caused by a Windows file-lock during autosave).
- `06_reward_shaping_experiment.md` - the central experimental finding: implementing
  potential-based reward shaping using the perfect solver's exact values, discovering
  that it degrades performance against an imperfect opponent (MCTS) while working exactly
  as intended against a perfect one, and why.
- `07_summary_and_open_questions.md` - a consolidated timeline of results and what's left
  to test.

Every number quoted in these documents (win rates, timings, state counts, sweep counts)
comes from an actual run performed during this project, not an estimate - the intent is
that this collection could be read on its own as a lab notebook.
